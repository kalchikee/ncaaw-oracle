// NCAAW Oracle v4.1 — NCAA Tournament Bracket Simulator
// Women's tournament: 68 teams (since 2022), Selection Monday (NOT Sunday)
// 10,000 simulations per round
// Uses Elo + AdjEM to compute game-level win probabilities

import { logger } from '../logger.js';
import { getDb } from '../db/database.js';
import { getAdjEM } from '../custom-adj-em/adjEmCalculator.js';
import { getTeamElo, eloWinProb, updateElo } from '../features/eloEngine.js';
import type { TournamentTeam, BracketSimResult } from '../types.js';

const N_SIMULATIONS = 10_000;

// WBB tournament bracket structure (68 teams, 4 regions)
// First Four: 4 games (play-in) → 64 teams in main bracket
// Rounds: R64 → R32 → S16 (Sweet 16) → E8 (Elite 8) → F4 → Championship

const REGIONS = ['Albany', 'Portland', 'Bridgeport', 'Spokane'];

// Seed matchups in first round (1 vs 16, 2 vs 15, etc.)
const FIRST_ROUND_MATCHUPS = [
  [1, 16], [8, 9], [5, 12], [4, 13],
  [6, 11], [3, 14], [7, 10], [2, 15],
];

// ─── Win probability calculator ───────────────────────────────────────────────

function tournamentWinProb(team1: TournamentTeam, team2: TournamentTeam): number {
  const eloDiff = team1.eloRating - team2.eloRating;
  const eloProb = eloWinProb(eloDiff);

  const adjEmDiff = team1.adjEM - team2.adjEM;
  const adjEmProb = 1 / (1 + Math.exp(-adjEmDiff / 12.0));

  // Blend: 60% AdjEM, 40% Elo in tournament (AdjEM more predictive late in season)
  return Math.max(0.05, Math.min(0.95, 0.60 * adjEmProb + 0.40 * eloProb));
}

// ─── Single-game simulation ───────────────────────────────────────────────────

function simulateGame(team1: TournamentTeam, team2: TournamentTeam): TournamentTeam {
  const prob = tournamentWinProb(team1, team2);
  return Math.random() < prob ? team1 : team2;
}

// ─── Single bracket simulation ────────────────────────────────────────────────

function simulateBracket(bracket: Map<string, TournamentTeam[]>): {
  finalFour: TournamentTeam[];
  champion: TournamentTeam;
} {
  const regionWinners: TournamentTeam[] = [];

  for (const region of REGIONS) {
    let teams = [...(bracket.get(region) ?? [])].sort((a, b) => a.seed - b.seed);
    if (teams.length === 0) continue;

    // Simulate through region: R64 → R32 → S16 → E8
    let round = [...teams];
    while (round.length > 1) {
      const nextRound: TournamentTeam[] = [];
      for (let i = 0; i < round.length; i += 2) {
        if (i + 1 >= round.length) { nextRound.push(round[i]); continue; }
        nextRound.push(simulateGame(round[i], round[i + 1]));
      }
      round = nextRound;
    }

    if (round.length > 0) regionWinners.push(round[0]);
  }

  // Final Four: Albany vs Bridgeport, Portland vs Spokane
  if (regionWinners.length < 4) {
    return { finalFour: regionWinners, champion: regionWinners[0] ?? createDummyTeam() };
  }

  const sf1 = simulateGame(regionWinners[0], regionWinners[2]);
  const sf2 = simulateGame(regionWinners[1], regionWinners[3]);
  const champion = simulateGame(sf1, sf2);

  return { finalFour: regionWinners, champion };
}

function createDummyTeam(): TournamentTeam {
  return {
    seed: 1, teamId: 'UNK', teamAbbr: 'UNK', teamName: 'Unknown',
    region: 'Unknown', adjEM: 0, eloRating: 1500,
    advancementProbs: {}, champProb: 0,
  };
}

// ─── Main bracket sim ─────────────────────────────────────────────────────────

export async function runBracketSim(nSims = N_SIMULATIONS): Promise<BracketSimResult> {
  logger.info({ nSims }, 'Running bracket simulation');

  // Load teams from DB or use seed data
  const teams = await loadTournamentTeams();

  if (teams.length === 0) {
    logger.warn('No tournament teams found — generating placeholder bracket');
    teams.push(...generatePlaceholderTeams());
  }

  // Organize by region
  const bracket = new Map<string, TournamentTeam[]>();
  for (const region of REGIONS) {
    bracket.set(region, teams.filter(t => t.region === region).sort((a, b) => a.seed - b.seed));
  }

  // Track advancement counts
  const advCounts = new Map<string, Record<string, number>>();
  const champCounts = new Map<string, number>();
  const f4Counts = new Map<string, number>();

  for (const team of teams) {
    advCounts.set(team.teamId, { R32: 0, S16: 0, E8: 0, F4: 0, Champ: 0 });
    champCounts.set(team.teamId, 0);
    f4Counts.set(team.teamId, 0);
  }

  // Run simulations
  for (let sim = 0; sim < nSims; sim++) {
    const { finalFour, champion } = simulateBracket(bracket);

    for (const t of finalFour) {
      f4Counts.set(t.teamId, (f4Counts.get(t.teamId) ?? 0) + 1);
      const counts = advCounts.get(t.teamId);
      if (counts) counts['F4']++;
    }

    const champCt = champCounts.get(champion.teamId) ?? 0;
    champCounts.set(champion.teamId, champCt + 1);
    const counts = advCounts.get(champion.teamId);
    if (counts) counts['Champ']++;
  }

  // Build results
  const resultTeams: TournamentTeam[] = teams.map(team => {
    const adv = advCounts.get(team.teamId) ?? {};
    const champCount = champCounts.get(team.teamId) ?? 0;
    return {
      ...team,
      advancementProbs: Object.fromEntries(
        Object.entries(adv).map(([round, count]) => [round, count / nSims])
      ),
      champProb: champCount / nSims,
    };
  });

  // Sort by champ probability for display
  const sortedByChamp = [...resultTeams].sort((a, b) => b.champProb - a.champProb);

  // Final Four
  const finalFourDisplay = sortedByChamp
    .sort((a, b) => (b.advancementProbs['F4'] ?? 0) - (a.advancementProbs['F4'] ?? 0))
    .slice(0, 8)
    .map(t => ({ team: t.teamName, prob: t.advancementProbs['F4'] ?? 0 }));

  const championshipDisplay = sortedByChamp.slice(0, 8).map(t => ({
    team: t.teamName, prob: t.champProb,
  }));

  // Cinderella watch: seeds 10–15 with >15% Sweet 16 probability
  const cinderellaWatch = resultTeams
    .filter(t => t.seed >= 10 && t.seed <= 15 && (t.advancementProbs['S16'] ?? 0) > 0.12)
    .sort((a, b) => (b.advancementProbs['S16'] ?? 0) - (a.advancementProbs['S16'] ?? 0))
    .slice(0, 6)
    .map(t => ({ team: t.teamName, seed: t.seed, sweetSixteenProb: t.advancementProbs['S16'] ?? 0 }));

  // Organize by region for embed
  const regionMap: Record<string, TournamentTeam[]> = {};
  for (const region of REGIONS) {
    regionMap[region] = resultTeams
      .filter(t => t.region === region)
      .sort((a, b) => a.seed - b.seed);
  }

  const year = new Date().getFullYear();

  logger.info({ teams: teams.length, simulations: nSims }, 'Bracket simulation complete');

  return {
    year,
    simulations: nSims,
    regions: regionMap,
    finalFour: finalFourDisplay,
    championship: championshipDisplay,
    cinderellaWatch,
  };
}

// ─── Load tournament teams ────────────────────────────────────────────────────

async function loadTournamentTeams(): Promise<TournamentTeam[]> {
  // In production: load from DB or ESPN tournament bracket API
  // For now, return empty to trigger placeholder
  return [];
}

// ─── Placeholder teams for testing ───────────────────────────────────────────

function generatePlaceholderTeams(): TournamentTeam[] {
  const placeholders: Array<{ name: string; abbr: string; adjEM: number; elo: number }> = [
    { name: 'South Carolina', abbr: 'SC', adjEM: 32, elo: 1845 },
    { name: 'UCLA', abbr: 'UCLA', adjEM: 28, elo: 1790 },
    { name: 'UConn', abbr: 'UCONN', adjEM: 27, elo: 1785 },
    { name: 'LSU', abbr: 'LSU', adjEM: 26, elo: 1780 },
    { name: 'Iowa', abbr: 'IOWA', adjEM: 24, elo: 1760 },
    { name: 'Texas', abbr: 'TEX', adjEM: 23, elo: 1750 },
    { name: 'Notre Dame', abbr: 'ND', adjEM: 22, elo: 1740 },
    { name: 'Stanford', abbr: 'STAN', adjEM: 21, elo: 1730 },
    { name: 'Tennessee', abbr: 'TENN', adjEM: 20, elo: 1720 },
    { name: 'Oklahoma', abbr: 'OKLA', adjEM: 19, elo: 1715 },
    { name: 'Kansas', abbr: 'KU', adjEM: 18, elo: 1700 },
    { name: 'Louisville', abbr: 'LOU', adjEM: 17, elo: 1695 },
    { name: 'Duke', abbr: 'DUKE', adjEM: 16, elo: 1690 },
    { name: 'North Carolina', abbr: 'UNC', adjEM: 15, elo: 1685 },
    { name: 'Oregon State', abbr: 'ORST', adjEM: 14, elo: 1680 },
    { name: 'Oregon', abbr: 'ORE', adjEM: 13, elo: 1675 },
  ];

  const teams: TournamentTeam[] = [];
  const seedsPerRegion = 4;

  for (let r = 0; r < REGIONS.length; r++) {
    for (let s = 1; s <= seedsPerRegion; s++) {
      const idx = r * seedsPerRegion + (s - 1);
      const pl = placeholders[idx % placeholders.length];
      const seed = s;
      teams.push({
        seed,
        teamId: `${pl.abbr}_${r}`,
        teamAbbr: pl.abbr,
        teamName: pl.name,
        region: REGIONS[r],
        adjEM: pl.adjEM - (seed - 1) * 2,
        eloRating: pl.elo - (seed - 1) * 30,
        advancementProbs: {},
        champProb: 0,
      });
    }
  }

  return teams;
}
