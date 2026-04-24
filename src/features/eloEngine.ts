// NCAAW Oracle v4.1 — Elo Rating Engine
// K-factor: 20 for regular season, 24 for conference tournaments, 28 for NCAA tournament
// Offseason regression: 50% toward 1500 + 25% recruiting Elo + 25% D1 mean
// WBB Elo range: ~1200 (bottom D-I) to ~1850 (top programs like SC, Iowa, LSU)

import { getDb } from '../db/database.js';
import { logger } from '../logger.js';

const D1_ELO_MEAN = 1500;
const K_REGULAR = 20;
const K_CONF_TOURNEY = 24;
const K_NCAA = 28;

// Pre-seeded Elo ratings for major programs (2025-26 season start)
// Based on final 2024-25 ratings with offseason regression applied
const SEED_ELOS: Record<string, number> = {
  // Power programs
  'SC': 1845,      // South Carolina
  'UCLA': 1790,
  'UCONN': 1785,
  'LSU': 1780,
  'IOWA': 1760,
  'TX': 1750,      // Texas
  'ND': 1740,      // Notre Dame
  'STAN': 1730,    // Stanford
  'TENN': 1720,    // Tennessee
  'OKLA': 1715,    // Oklahoma
  'KANS': 1700,    // Kansas
  'LOU': 1695,     // Louisville
  'DUKE': 1690,
  'NC': 1685,      // North Carolina
  'ORST': 1680,    // Oregon State
  'ORE': 1675,     // Oregon
  'COLO': 1670,    // Colorado
  'MISS': 1660,    // Ole Miss
  'ARK': 1655,     // Arkansas
  'VAND': 1650,    // Vanderbilt
  'MICH': 1640,    // Michigan
  'MSU': 1635,     // Michigan State / Mississippi State
  'FLOR': 1630,    // Florida
  'MARQ': 1625,    // Marquette
  'VLNV': 1620,    // Villanova
  'GBAY': 1600,    // Green Bay (mid-major example)
  'STF': 1580,
};

export function seedElos(): void {
  try {
    const db = getDb();
    const existing = db.prepare('SELECT COUNT(*) as count FROM elo_ratings').getAsObject();
    if (Number(existing['count']) > 0) return;

    const stmt = db.prepare('INSERT OR IGNORE INTO elo_ratings (team, elo, last_updated) VALUES (?, ?, ?)');
    const now = new Date().toISOString();
    for (const [team, elo] of Object.entries(SEED_ELOS)) {
      stmt.run([team, elo, now]);
    }
    stmt.free();
    logger.info({ teams: Object.keys(SEED_ELOS).length }, 'Elo ratings seeded');
  } catch (err) {
    logger.error({ err }, 'Failed to seed Elo ratings');
  }
}

export function getTeamElo(teamAbbr: string): number {
  try {
    const db = getDb();
    const stmt = db.prepare('SELECT elo FROM elo_ratings WHERE team = ?');
    stmt.bind([teamAbbr]);
    if (stmt.step()) {
      const result = stmt.getAsObject();
      stmt.free();
      return Number(result['elo'] ?? D1_ELO_MEAN);
    }
    stmt.free();
    return D1_ELO_MEAN;
  } catch {
    return D1_ELO_MEAN;
  }
}

export function getEloDiff(homeAbbr: string, awayAbbr: string): number {
  return getTeamElo(homeAbbr) - getTeamElo(awayAbbr);
}

// Win probability from Elo difference (logistic function)
export function eloWinProb(eloDiff: number): number {
  return 1 / (1 + Math.pow(10, -eloDiff / 400));
}

// Log5 win probability from two win percentages
export function log5Prob(homeWinPct: number, awayWinPct: number): number {
  const h = Math.max(0.01, Math.min(0.99, homeWinPct));
  const a = Math.max(0.01, Math.min(0.99, awayWinPct));
  return (h * (1 - a)) / (h * (1 - a) + a * (1 - h));
}

// Update Elo after a game result
export function updateElo(
  homeAbbr: string,
  awayAbbr: string,
  homeWon: boolean,
  gameType: 'regular' | 'conf_tourney' | 'ncaa' = 'regular',
  marginOfVictory?: number
): void {
  const homeElo = getTeamElo(homeAbbr);
  const awayElo = getTeamElo(awayAbbr);

  const expectedHome = eloWinProb(homeElo - awayElo + 100); // +100 for home court
  const actualHome = homeWon ? 1 : 0;

  let K: number;
  switch (gameType) {
    case 'ncaa': K = K_NCAA; break;
    case 'conf_tourney': K = K_CONF_TOURNEY; break;
    default: K = K_REGULAR;
  }

  // Margin of victory multiplier (capped to avoid extreme shifts)
  let movMult = 1.0;
  if (marginOfVictory !== undefined) {
    const mov = Math.abs(marginOfVictory);
    // WBB: larger MOV multiplier since games can be very lopsided
    movMult = Math.log(Math.max(mov, 1) + 1) / Math.log(5);
    movMult = Math.max(0.5, Math.min(2.0, movMult));
  }

  const delta = K * movMult * (actualHome - expectedHome);
  const newHomeElo = Math.round(homeElo + delta);
  const newAwayElo = Math.round(awayElo - delta);

  updateTeamElo(homeAbbr, newHomeElo);
  updateTeamElo(awayAbbr, newAwayElo);
}

function updateTeamElo(teamAbbr: string, newElo: number): void {
  try {
    const db = getDb();
    const now = new Date().toISOString();
    const stmt = db.prepare(
      'INSERT INTO elo_ratings (team, elo, last_updated) VALUES (?, ?, ?) ' +
      'ON CONFLICT(team) DO UPDATE SET elo = excluded.elo, last_updated = excluded.last_updated'
    );
    stmt.run([teamAbbr, newElo, now]);
    stmt.free();
  } catch (err) {
    logger.error({ err, teamAbbr }, 'Failed to update Elo');
  }
}

// Preseason Elo regression (offseason reset)
// new_season_elo = 0.50 × prior_final + 0.25 × recruiting_elo + 0.25 × D1_mean
export function applyOffseasonRegression(teamAbbr: string, recruitingElo?: number): void {
  const currentElo = getTeamElo(teamAbbr);
  const recElo = recruitingElo ?? D1_ELO_MEAN;
  const newElo = Math.round(0.50 * currentElo + 0.25 * recElo + 0.25 * D1_ELO_MEAN);
  updateTeamElo(teamAbbr, newElo);
}
