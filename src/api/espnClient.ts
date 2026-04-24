// NCAAW Oracle v4.1 — ESPN / NCAA Data Client
// Fetches: schedules, scores, team stats, box scores, player stats
// Women's D-I basketball only — 364 teams, ~5,500 games/season

import fetch from 'node-fetch';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';
import type { WBBGame, WBBTeam, WBBPlayer } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const CACHE_DIR = resolve(__dirname, '../../cache');
mkdirSync(CACHE_DIR, { recursive: true });

const ESPN_BASE = 'https://site.api.espn.com/apis/site/v2/sports/basketball/womens-college-basketball';
const ESPN_CORE = 'https://sports.core.api.espn.com/v2/sports/basketball/leagues/womens-college-basketball';
const CACHE_TTL_MS = 3 * 60 * 60 * 1000; // 3 hours

// ─── Cache helpers ────────────────────────────────────────────────────────────

function cachePath(key: string): string {
  return resolve(CACHE_DIR, `${key.replace(/[^a-z0-9_-]/gi, '_')}.json`);
}

function readCache<T>(key: string): T | null {
  const p = cachePath(key);
  if (!existsSync(p)) return null;
  try {
    const { ts, data } = JSON.parse(readFileSync(p, 'utf8'));
    if (Date.now() - ts > CACHE_TTL_MS) return null;
    return data as T;
  } catch {
    return null;
  }
}

function writeCache(key: string, data: unknown): void {
  writeFileSync(cachePath(key), JSON.stringify({ ts: Date.now(), data }));
}

// ─── Retry fetch ──────────────────────────────────────────────────────────────

async function fetchWithRetry(url: string, retries = 3): Promise<unknown> {
  for (let i = 0; i < retries; i++) {
    try {
      const resp = await fetch(url, {
        headers: { 'User-Agent': 'NCAAW-Oracle/4.1' },
        signal: AbortSignal.timeout(15000),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
      return await resp.json();
    } catch (err) {
      if (i === retries - 1) throw err;
      const wait = 1000 * (i + 1);
      logger.warn({ url, attempt: i + 1, wait }, 'Fetch failed, retrying...');
      await new Promise(r => setTimeout(r, wait));
    }
  }
}

// ─── Schedule fetch ───────────────────────────────────────────────────────────

export async function fetchSchedule(dateStr: string, forceRefresh = false): Promise<WBBGame[]> {
  const key = `schedule_${dateStr}`;
  if (!forceRefresh) {
    const cached = readCache<WBBGame[]>(key);
    if (cached) { logger.debug({ dateStr, count: cached.length }, 'Schedule from cache'); return cached; }
  }

  const compact = dateStr.replace(/-/g, '');
  const url = `${ESPN_BASE}/scoreboard?dates=${compact}&limit=200&groups=50`;

  try {
    const data = await fetchWithRetry(url) as Record<string, unknown>;
    const events = (data.events as unknown[]) ?? [];
    const games: WBBGame[] = events.map(parseEvent).filter(Boolean) as WBBGame[];
    writeCache(key, games);
    logger.info({ dateStr, count: games.length }, 'Schedule fetched from ESPN');
    return games;
  } catch (err) {
    logger.error({ err, dateStr }, 'Failed to fetch schedule');
    return [];
  }
}

function parseEvent(event: unknown): WBBGame | null {
  const e = event as Record<string, unknown>;
  const competition = (e.competitions as Record<string, unknown>[])?.[0];
  if (!competition) return null;

  const competitors = competition.competitors as Record<string, unknown>[];
  const home = competitors?.find((c: Record<string, unknown>) => c.homeAway === 'home');
  const away = competitors?.find((c: Record<string, unknown>) => c.homeAway === 'away');
  if (!home || !away) return null;

  const homeTeam = home.team as Record<string, unknown>;
  const awayTeam = away.team as Record<string, unknown>;
  const status = (e.status as Record<string, unknown>)?.type as Record<string, unknown>;

  const neutralSite = Boolean(competition.neutralSite);
  const venue = competition.venue as Record<string, unknown> | undefined;

  return {
    gameId: String(e.id ?? ''),
    gameDate: String(e.date ?? '').split('T')[0],
    gameTime: String(e.date ?? ''),
    status: String(status?.name ?? 'Scheduled'),
    homeTeam: {
      teamId: String(homeTeam?.id ?? ''),
      teamAbbr: String(homeTeam?.abbreviation ?? '').toUpperCase(),
      teamName: String(homeTeam?.displayName ?? ''),
      score: home.score != null ? Number(home.score) : undefined,
      seed: home.curatedRank != null ? Number((home.curatedRank as Record<string, unknown>)?.current) : undefined,
    },
    awayTeam: {
      teamId: String(awayTeam?.id ?? ''),
      teamAbbr: String(awayTeam?.abbreviation ?? '').toUpperCase(),
      teamName: String(awayTeam?.displayName ?? ''),
      score: away.score != null ? Number(away.score) : undefined,
      seed: away.curatedRank != null ? Number((away.curatedRank as Record<string, unknown>)?.current) : undefined,
    },
    neutralSite,
    arena: venue ? String((venue.fullName as string) ?? '') : undefined,
    arenaCity: venue ? String(((venue.address as Record<string, unknown>)?.city as string) ?? '') : undefined,
  };
}

// ─── Team stats ───────────────────────────────────────────────────────────────

export async function fetchAllTeamStats(season = '2025-26', forceRefresh = false): Promise<Map<string, WBBTeam>> {
  const key = `team_stats_${season}`;
  if (!forceRefresh) {
    const cached = readCache<WBBTeam[]>(key);
    if (cached) {
      const m = new Map<string, WBBTeam>();
      cached.forEach(t => m.set(t.teamId, t));
      return m;
    }
  }

  const url = `${ESPN_BASE}/teams?limit=400&season=${season.replace('-', '')}`;
  try {
    const data = await fetchWithRetry(url) as Record<string, unknown>;
    const sports = (data.sports as Record<string, unknown>[])?.[0];
    const leagues = (sports?.leagues as Record<string, unknown>[])?.[0];
    const teams = (leagues?.teams as Record<string, unknown>[]) ?? [];

    const result = new Map<string, WBBTeam>();
    for (const t of teams) {
      const team = t.team as Record<string, unknown>;
      if (!team) continue;
      const wbbTeam = await fetchTeamDetails(String(team.id), season);
      if (wbbTeam) result.set(wbbTeam.teamId, wbbTeam);
    }

    writeCache(key, [...result.values()]);
    logger.info({ season, count: result.size }, 'Team stats fetched');
    return result;
  } catch (err) {
    logger.error({ err, season }, 'Failed to fetch team stats');
    return new Map();
  }
}

export async function fetchTeamDetails(teamId: string, season = '2025-26'): Promise<WBBTeam | null> {
  const key = `team_detail_${teamId}_${season}`;
  const cached = readCache<WBBTeam>(key);
  if (cached) return cached;

  const url = `${ESPN_BASE}/teams/${teamId}?enable=roster,projection,stats&season=${season.replace('-', '')}`;
  try {
    const data = await fetchWithRetry(url) as Record<string, unknown>;
    const t = data.team as Record<string, unknown>;
    if (!t) return null;

    const record = (t.record as Record<string, unknown>)?.items as Record<string, unknown>[];
    const overallRecord = record?.find((r: Record<string, unknown>) => r.type === 'total');
    const stats = (t.statistics as Record<string, unknown>)?.splits?.categories as Record<string, unknown>[];

    const getStatValue = (cats: Record<string, unknown>[] | undefined, catName: string, statName: string): number => {
      const cat = cats?.find((c: Record<string, unknown>) => c.name === catName);
      const stat = (cat?.stats as Record<string, unknown>[])?.find((s: Record<string, unknown>) => s.name === statName);
      return Number(stat?.value ?? 0);
    };

    const wins = Number(overallRecord?.wins ?? 0);
    const losses = Number(overallRecord?.losses ?? 0);
    const winPct = wins + losses > 0 ? wins / (wins + losses) : 0.5;

    // WBB Four Factors from ESPN stats
    const efgPct = getStatValue(stats, 'offensive', 'effectiveFieldGoalPct') || 0.45;
    const tovPct = getStatValue(stats, 'general', 'turnoverPct') || 0.21;
    const orbPct = getStatValue(stats, 'rebounds', 'offensiveReboundPct') || 0.31;
    const ftRate = getStatValue(stats, 'offensive', 'freeThrowRate') || 0.30;
    const threePtPct = getStatValue(stats, 'offensive', 'threePointPct') || 0.31;
    const twoPtPct = getStatValue(stats, 'offensive', 'twoPointPct') || 0.46;

    // Points per game → estimate efficiency (WBB avg ~62 PPG)
    const ppg = getStatValue(stats, 'offensive', 'avgPoints') || 62.0;
    const ppga = getStatValue(stats, 'defensive', 'avgPoints') || 62.0;
    const tempo = getStatValue(stats, 'general', 'possessionsPerGame') || 66.0;

    // Raw efficiency: pts per 100 possessions
    const rawOE = tempo > 0 ? (ppg / tempo) * 100 : 93.5;
    const rawDE = tempo > 0 ? (ppga / tempo) * 100 : 93.5;

    // Pythagorean (exp = 10)
    const pythagoreanWinPct = rawOE ** 10 / (rawOE ** 10 + rawDE ** 10);

    const team: WBBTeam = {
      teamId,
      teamName: String(t.displayName ?? ''),
      teamAbbr: String(t.abbreviation ?? '').toUpperCase(),
      conference: String((t.conference as Record<string, unknown>)?.abbreviation ?? ''),
      wins,
      losses,
      winPct,
      adjOE: rawOE,    // will be updated by custom AdjEM calculator
      adjDE: rawDE,
      adjEM: rawOE - rawDE,
      adjTempo: tempo,
      efgPct,
      tovPct,
      orbPct,
      ftRate,
      threePtPct,
      twoPtPct,
      blockPct: getStatValue(stats, 'defensive', 'blockPct') || 0.10,
      stealPct: getStatValue(stats, 'defensive', 'stealPct') || 0.12,
      pythagoreanWinPct,
      gamesPlayed: wins + losses,
    };

    writeCache(key, team);
    return team;
  } catch (err) {
    logger.debug({ err, teamId }, 'Failed to fetch team detail');
    return null;
  }
}

// ─── Team last game date ──────────────────────────────────────────────────────

export async function fetchTeamLastGameDate(teamAbbr: string, beforeDate: string): Promise<string | null> {
  const key = `last_game_${teamAbbr}_${beforeDate}`;
  const cached = readCache<string>(key);
  if (cached) return cached;

  // Look back 30 days from beforeDate
  const before = new Date(beforeDate);
  const after = new Date(beforeDate);
  after.setDate(after.getDate() - 30);

  const afterStr = after.toISOString().split('T')[0].replace(/-/g, '');
  const beforeStr = before.toISOString().split('T')[0].replace(/-/g, '');

  const url = `${ESPN_BASE}/scoreboard?dates=${afterStr}-${beforeStr}&limit=500&groups=50`;
  try {
    const data = await fetchWithRetry(url) as Record<string, unknown>;
    const events = (data.events as Record<string, unknown>[]) ?? [];

    let lastDate: string | null = null;
    for (const evt of events) {
      const comp = (evt.competitions as Record<string, unknown>[])?.[0];
      const competitors = comp?.competitors as Record<string, unknown>[];
      const isTeam = competitors?.some((c: Record<string, unknown>) =>
        String((c.team as Record<string, unknown>)?.abbreviation ?? '').toUpperCase() === teamAbbr.toUpperCase()
      );
      if (isTeam) {
        const evtDate = String(evt.date ?? '').split('T')[0];
        if (evtDate < beforeDate && (!lastDate || evtDate > lastDate)) {
          lastDate = evtDate;
        }
      }
    }

    writeCache(key, lastDate);
    return lastDate;
  } catch {
    return null;
  }
}

// ─── Player stats ─────────────────────────────────────────────────────────────

export async function fetchTeamRoster(teamId: string, season = '2025-26'): Promise<WBBPlayer[]> {
  const key = `roster_${teamId}_${season}`;
  const cached = readCache<WBBPlayer[]>(key);
  if (cached) return cached;

  const url = `${ESPN_BASE}/teams/${teamId}/roster?season=${season.replace('-', '')}`;
  try {
    const data = await fetchWithRetry(url) as Record<string, unknown>;
    const athletes = (data.athletes as Record<string, unknown>[]) ?? [];

    const players: WBBPlayer[] = athletes.map(a => ({
      playerId: String(a.id ?? ''),
      playerName: String(a.displayName ?? ''),
      teamId,
      teamAbbr: '',
      position: String((a.position as Record<string, unknown>)?.abbreviation ?? ''),
      minutesPerGame: Number((a.statistics as Record<string, unknown>)?.minutes ?? 0),
      usageRate: Number((a.statistics as Record<string, unknown>)?.usageRate ?? 0.20),
      bpm: Number((a.statistics as Record<string, unknown>)?.boxPlusMinus ?? 0),
      offBpm: Number((a.statistics as Record<string, unknown>)?.offensiveBPM ?? 0),
      defBpm: Number((a.statistics as Record<string, unknown>)?.defensiveBPM ?? 0),
    }));

    writeCache(key, players);
    return players;
  } catch {
    return [];
  }
}

// ─── Completed game results ───────────────────────────────────────────────────

export async function fetchCompletedGames(dateStr: string): Promise<Array<{
  gameId: string; homeTeam: string; awayTeam: string;
  homeScore: number; awayScore: number;
}>> {
  const key = `results_${dateStr}`;
  const cached = readCache<Array<{ gameId: string; homeTeam: string; awayTeam: string; homeScore: number; awayScore: number }>>(key);
  if (cached) return cached;

  const games = await fetchSchedule(dateStr, true);
  const results = games
    .filter(g => g.status.toLowerCase().includes('final'))
    .map(g => ({
      gameId: g.gameId,
      homeTeam: g.homeTeam.teamAbbr,
      awayTeam: g.awayTeam.teamAbbr,
      homeScore: g.homeTeam.score ?? 0,
      awayScore: g.awayTeam.score ?? 0,
    }));

  writeCache(key, results);
  return results;
}

// ─── Default team (fallback) ──────────────────────────────────────────────────

export function defaultWBBTeam(abbr: string): WBBTeam {
  return {
    teamId: abbr,
    teamName: abbr,
    teamAbbr: abbr,
    conference: 'Unknown',
    wins: 0,
    losses: 0,
    winPct: 0.5,
    adjOE: 93.5,   // WBB D1 average (~62 PPG at ~66 poss)
    adjDE: 93.5,
    adjEM: 0.0,
    adjTempo: 66.0,
    efgPct: 0.45,
    tovPct: 0.21,
    orbPct: 0.31,
    ftRate: 0.30,
    threePtPct: 0.31,
    twoPtPct: 0.46,
    blockPct: 0.10,
    stealPct: 0.12,
    pythagoreanWinPct: 0.5,
    gamesPlayed: 0,
  };
}

// ─── AP Top 25 rankings ───────────────────────────────────────────────────────

export interface RankedTeam {
  rank: number;
  teamId: string;
  teamAbbr: string;
  teamName: string;
}

// Cache rankings for 6 hours (polls update weekly, but this avoids stale data)
const RANKINGS_TTL_MS = 6 * 60 * 60 * 1000;

export async function fetchTop25(forceRefresh = false): Promise<RankedTeam[]> {
  const key = 'top25_rankings';
  if (!forceRefresh) {
    const cached = readCache<RankedTeam[]>(key);
    if (cached && cached.length > 0) {
      logger.debug({ count: cached.length }, 'Top 25 from cache');
      return cached;
    }
  }

  const url = `${ESPN_BASE}/rankings`;
  try {
    const data = await fetchWithRetry(url) as Record<string, unknown>;
    const rankings = (data.rankings as Record<string, unknown>[]) ?? [];

    // ESPN returns multiple poll types — prefer AP, fallback to first available
    const apPoll = rankings.find((r: Record<string, unknown>) =>
      String(r.name ?? '').toLowerCase().includes('ap')
    ) ?? rankings[0];

    if (!apPoll) {
      logger.warn('No rankings found from ESPN — Top 25 filter unavailable');
      return [];
    }

    const ranks = (apPoll.ranks as Record<string, unknown>[]) ?? [];
    const top25: RankedTeam[] = ranks.slice(0, 25).map((entry: Record<string, unknown>) => {
      const team = entry.team as Record<string, unknown>;
      return {
        rank: Number(entry.current ?? entry.rank ?? 0),
        teamId: String(team?.id ?? ''),
        teamAbbr: String(team?.abbreviation ?? '').toUpperCase(),
        teamName: String(team?.displayName ?? ''),
      };
    });

    writeCache(key, top25);
    logger.info({ count: top25.length }, 'Top 25 rankings fetched from ESPN');
    return top25;
  } catch (err) {
    logger.warn({ err }, 'Failed to fetch Top 25 rankings — no filter applied');
    return [];
  }
}

// Returns a Set of teamIds in the current Top 25
export async function getTop25TeamIds(forceRefresh = false): Promise<Set<string>> {
  const ranked = await fetchTop25(forceRefresh);
  return new Set(ranked.map(t => t.teamId));
}

// ─── Season helper ────────────────────────────────────────────────────────────

export function getCurrentWBBSeason(): string {
  const now = new Date();
  const year = now.getFullYear();
  const month = now.getMonth() + 1;
  // WBB season spans two calendar years; if before July treat as prior season
  if (month >= 7) return `${year}-${String(year + 1).slice(2)}`;
  return `${year - 1}-${String(year).slice(2)}`;
}
