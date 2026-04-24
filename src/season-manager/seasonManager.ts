// NCAAW Oracle v4.1 — Season Lifecycle Manager
// Women's Basketball: Early November → NCAA Championship (early April)
// Phases: Dormant | PreseasonSetup | EarlySeason | ConferencePlay | ConferenceTournaments | NCAATorunament | SeasonEnd

export type SeasonPhase =
  | 'dormant'
  | 'preseason_setup'
  | 'early_season'
  | 'conference_play'
  | 'conference_tournaments'
  | 'ncaa_tournament'
  | 'season_end';

export interface SeasonStatus {
  phase: SeasonPhase;
  isActive: boolean;
  isEarlySeason: boolean;
  isTournament: boolean;
  label: string;
  year: number;
  selectionMonday?: string;   // YYYY-MM-DD of Selection Monday
  championshipDate?: string;  // YYYY-MM-DD of Championship
}

// Approximate season dates (update annually via config/season.json)
// 2025–26 season
const SEASON_DATES = {
  PRESEASON_START: '2025-10-01',   // October — preseason setup
  EARLY_SEASON_START: '2025-11-01', // Early November — games begin
  CONFERENCE_PLAY_START: '2026-01-01', // ~Jan 1
  CONF_TOURNAMENTS_START: '2026-03-04', // early March
  NCAA_SELECTION_MONDAY: '2026-03-16',  // Selection Monday
  NCAA_CHAMPIONSHIP: '2026-04-05',      // Women's championship Sunday
  SEASON_END: '2026-04-07',            // day after championship
};

export function getSeasonStatus(dateStr?: string): SeasonStatus {
  const now = dateStr ? new Date(dateStr) : new Date();
  const year = now.getFullYear();

  const preseasonStart = new Date(SEASON_DATES.PRESEASON_START);
  const earlySeasonStart = new Date(SEASON_DATES.EARLY_SEASON_START);
  const confPlayStart = new Date(SEASON_DATES.CONFERENCE_PLAY_START);
  const confTourneysStart = new Date(SEASON_DATES.CONF_TOURNAMENTS_START);
  const selectionMonday = new Date(SEASON_DATES.NCAA_SELECTION_MONDAY);
  const championship = new Date(SEASON_DATES.NCAA_CHAMPIONSHIP);
  const seasonEnd = new Date(SEASON_DATES.SEASON_END);

  let phase: SeasonPhase;

  if (now >= seasonEnd) {
    phase = 'season_end';
  } else if (now >= selectionMonday) {
    phase = 'ncaa_tournament';
  } else if (now >= confTourneysStart) {
    phase = 'conference_tournaments';
  } else if (now >= confPlayStart) {
    phase = 'conference_play';
  } else if (now >= earlySeasonStart) {
    phase = 'early_season';
  } else if (now >= preseasonStart) {
    phase = 'preseason_setup';
  } else {
    phase = 'dormant';
  }

  const isActive = !['dormant'].includes(phase);
  const isEarlySeason = phase === 'early_season';
  const isTournament = phase === 'ncaa_tournament';

  const labels: Record<SeasonPhase, string> = {
    dormant: 'Dormant (off-season)',
    preseason_setup: 'Preseason Setup',
    early_season: 'Early Season',
    conference_play: 'Conference Play',
    conference_tournaments: 'Conference Tournaments',
    ncaa_tournament: 'NCAA Tournament',
    season_end: 'Season Complete',
  };

  return {
    phase,
    isActive,
    isEarlySeason,
    isTournament,
    label: labels[phase],
    year,
    selectionMonday: SEASON_DATES.NCAA_SELECTION_MONDAY,
    championshipDate: SEASON_DATES.NCAA_CHAMPIONSHIP,
  };
}

// Check if today is Selection Monday
export function isSelectionMonday(dateStr?: string): boolean {
  const date = dateStr ?? new Date().toISOString().split('T')[0];
  return date === SEASON_DATES.NCAA_SELECTION_MONDAY;
}

// Check if today is the day after championship (trigger final summary)
export function isSeasonEndDay(dateStr?: string): boolean {
  const date = dateStr ?? new Date().toISOString().split('T')[0];
  return date >= SEASON_DATES.SEASON_END;
}

// Get bootstrap blend weight based on games played
// Returns [priorWeight, inSeasonWeight]
export function getBootstrapBlendWeights(gamesPlayed: number): [number, number] {
  if (gamesPlayed <= 2) return [0.80, 0.20];
  if (gamesPlayed <= 5) return [0.60, 0.40];
  if (gamesPlayed <= 10) return [0.40, 0.60];
  if (gamesPlayed <= 15) return [0.25, 0.75];
  if (gamesPlayed <= 20) return [0.15, 0.85];
  return [0.05, 0.95];
}

// Is a team in bootstrap territory (< 6 games)?
export function isBootstrapTeam(gamesPlayed: number): boolean {
  return gamesPlayed <= 5;
}

// Get early-season label for embeds
export function getEarlySeasonLabel(gamesPlayedHome: number, gamesPlayedAway: number): string | null {
  if (gamesPlayedHome <= 5 || gamesPlayedAway <= 5) return 'EARLY SEASON';
  if (gamesPlayedHome <= 10 || gamesPlayedAway <= 10) return 'BLENDING';
  return null;
}

// Tournament round display names
export const TOURNAMENT_ROUNDS: Record<string, string> = {
  R68: 'First Four',
  R64: 'First Round',
  R32: 'Second Round',
  S16: 'Sweet Sixteen',
  E8: 'Elite Eight',
  F4: 'Final Four',
  Champ: 'Championship',
};
