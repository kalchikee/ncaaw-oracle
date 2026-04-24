// NCAAW Oracle v4.1 — Core Type Definitions
// Women's Basketball specific types — DO NOT mix with men's data

// ─── WBB Team stats ───────────────────────────────────────────────────────────

export interface WBBTeam {
  teamId: string;           // ESPN team ID
  teamName: string;
  teamAbbr: string;
  conference: string;
  // Season record
  wins: number;
  losses: number;
  winPct: number;
  // Adjusted efficiency (custom-computed or Her Hoop Stats)
  adjOE: number;            // adjusted offensive efficiency (pts/100 poss)
  adjDE: number;            // adjusted defensive efficiency (pts/100 poss)
  adjEM: number;            // adjOE - adjDE (higher = better)
  adjTempo: number;         // adjusted possessions per 40 min
  // Women's Four Factors (WBB averages: eFG% ~45%, TOV% ~21%, ORB% ~31%)
  efgPct: number;           // effective FG% (WBB avg ~0.45)
  tovPct: number;           // turnover % (WBB avg ~0.21)
  orbPct: number;           // offensive rebound rate (WBB avg ~0.31)
  ftRate: number;           // FTA/FGA
  threePtPct: number;       // 3-point % (WBB avg ~0.30-0.32)
  twoPtPct: number;         // 2-point %
  blockPct: number;         // block %
  stealPct: number;         // steal %
  // Pythagorean
  pythagoreanWinPct: number;
  // Games played (for bootstrap blending)
  gamesPlayed: number;
  // Prior-year data (for bootstrap)
  priorAdjEM?: number;
  priorAdjOE?: number;
  priorAdjDE?: number;
  priorTempo?: number;
  // Home/Away splits
  homeWins?: number;
  homeLosses?: number;
  awayWins?: number;
  awayLosses?: number;
  // Last game date
  lastGameDate?: string;
  // SOS
  strengthOfSchedule?: number;
  // Conference quality rank
  conferenceRank?: number;
}

export interface WBBPlayer {
  playerId: string;
  playerName: string;
  teamId: string;
  teamAbbr: string;
  position: string;
  minutesPerGame: number;
  usageRate: number;       // usage % (higher in WBB for stars)
  bpm: number;             // Box Plus/Minus
  offBpm: number;
  defBpm: number;
  // Availability
  injured?: boolean;
  injuryStatus?: string;   // 'Out' | 'Doubtful' | 'Questionable' | 'Probable'
  // Draft/transfer
  wnnaDraftProspect?: boolean;   // projected WNBA draft pick
  transferPortalStatus?: string; // 'staying' | 'portal' | 'graduated'
}

export interface WBBGame {
  gameId: string;
  gameDate: string;        // YYYY-MM-DD
  gameTime?: string;       // ISO datetime
  status: string;          // 'Scheduled' | 'Live' | 'Final'
  homeTeam: WBBGameTeam;
  awayTeam: WBBGameTeam;
  neutralSite: boolean;
  arena?: string;
  arenaCity?: string;
  conference?: string;     // conference game?
  tournamentRound?: string; // 'R64' | 'R32' | 'S16' | 'E8' | 'F4' | 'Champ'
}

export interface WBBGameTeam {
  teamId: string;
  teamAbbr: string;
  teamName: string;
  score?: number;
  seed?: number;           // tournament seed
}

// ─── Feature vector (40+ WBB-calibrated features) ────────────────────────────

export interface FeatureVector {
  // Identity
  game_id: string;
  home_team: string;
  away_team: string;
  game_date: string;
  // Early season bootstrap flag
  early_season_home: boolean;
  early_season_away: boolean;
  blend_weight_home: number;   // 0–1, prior-year weight
  blend_weight_away: number;

  // ── Team strength (bootstrapped if early season) ──────────────────────────
  elo_diff: number;            // home Elo - away Elo
  adj_em_diff: number;         // home AdjEM - away AdjEM
  adj_oe_diff: number;         // home AdjOE - away AdjOE
  adj_de_diff: number;         // home AdjDE - away AdjDE
  adj_tempo_diff: number;      // home tempo - away tempo
  pythagorean_diff: number;    // home pythagorean winPct - away

  // ── Four Factors (WBB norms: eFG%~45%, TOV%~21%, ORB%~31%) ──────────────
  efg_pct_diff: number;
  tov_pct_diff: number;        // negative = home turns over less (better)
  oreb_pct_diff: number;
  ft_rate_diff: number;
  three_pt_pct_diff: number;
  two_pt_pct_diff: number;
  block_pct_diff: number;
  steal_pct_diff: number;

  // ── Talent / continuity (bootstrapped) ───────────────────────────────────
  recruiting_composite_diff: number;
  returning_minutes_diff: number;
  portal_impact_diff: number;
  star_player_impact_diff: number;  // top player BPM × usage delta
  experience_diff: number;          // avg class year (5th-year seniors matter)

  // ── Context ───────────────────────────────────────────────────────────────
  sos_diff: number;
  conference_quality_diff: number;
  bench_depth_diff: number;

  // ── Venue ─────────────────────────────────────────────────────────────────
  is_home: number;             // 1 = true home game
  is_neutral_site: number;     // 1 = neutral site

  // ── Fatigue / rest ────────────────────────────────────────────────────────
  rest_days_diff: number;      // home rest days - away rest days

  // ── Star player availability ──────────────────────────────────────────────
  star_available_home: number; // 1 = star player is available
  star_available_away: number;
  star_injury_penalty_home: number;  // AdjEM penalty for missing star
  star_injury_penalty_away: number;

  // ── Market edge ───────────────────────────────────────────────────────────
  has_line: boolean;
  vegas_home_prob?: number;    // implied prob from spread/ML
  model_edge?: number;         // model_prob - vegas_prob

  // ── Monte Carlo outputs (filled by MC engine) ─────────────────────────────
  mc_home_win_pct?: number;
  mc_expected_home_pts?: number;
  mc_expected_away_pts?: number;
  mc_expected_spread?: number;
}

// ─── Monte Carlo result ───────────────────────────────────────────────────────

export interface MonteCarloResult {
  homeWinPct: number;
  awayWinPct: number;
  expectedHomeScore: number;
  expectedAwayScore: number;
  expectedSpread: number;      // home - away
  otProb: number;
  simulations: number;
}

// ─── Prediction record ────────────────────────────────────────────────────────

export interface Prediction {
  game_id: string;
  game_date: string;
  home_team: string;
  away_team: string;
  mc_prob: number;             // raw Monte Carlo home win prob
  calibrated_prob: number;     // Platt-scaled probability
  confidence_tier: ConfidenceTier;
  expected_home_score: number;
  expected_away_score: number;
  expected_spread: number;
  has_line: boolean;
  vegas_spread?: number;
  vegas_ml_home?: number;
  model_edge?: number;
  edge_tier?: EdgeTier;
  early_season: boolean;
  feature_vector: Partial<FeatureVector>;
  model_version: string;
  created_at: string;
  // Result (filled after game)
  actual_home_score?: number;
  actual_away_score?: number;
  correct?: boolean;
  covered_spread?: boolean;
}

export type ConfidenceTier =
  | 'coin_flip'        // 50–58%
  | 'moderate'         // 58–65%
  | 'strong'           // 65–74%
  | 'high_conviction'  // 74–80%
  | 'extreme';         // 80%+

export type EdgeTier =
  | 'agreement'        // <4%
  | 'small'            // 4–8%
  | 'meaningful'       // 8–12%
  | 'large';           // >12%

// ─── DB types ─────────────────────────────────────────────────────────────────

export interface GameResult {
  game_id: string;
  game_date: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  home_won: boolean;
}

export interface AccuracyLog {
  date: string;
  total_games: number;
  correct: number;
  accuracy: number;
  high_conviction_total: number;
  high_conviction_correct: number;
  high_conviction_accuracy: number;
  extreme_total: number;
  extreme_correct: number;
  ats_total: number;
  ats_correct: number;
  brier_score: number;
}

export interface SeasonRecord {
  total: number;
  correct: number;
  hc_total: number;
  hc_correct: number;
  extreme_total: number;
  extreme_correct: number;
  ats_total: number;
  ats_correct: number;
  brier_sum: number;
  games_with_brier: number;
  week_records: WeekRecord[];
}

export interface WeekRecord {
  week: number;
  start_date: string;
  total: number;
  correct: number;
  accuracy: number;
}

export interface EloRating {
  team: string;
  elo: number;
  last_updated: string;
}

// ─── Pipeline options ─────────────────────────────────────────────────────────

export interface PipelineOptions {
  date?: string;
  verbose?: boolean;
  forceRefresh?: boolean;
  alertMode?: 'morning' | 'recap' | 'weekly' | 'bracket' | 'preseason' | null;
}

// ─── Custom AdjEM ─────────────────────────────────────────────────────────────

export interface AdjEfficiency {
  teamId: string;
  teamName: string;
  adjOE: number;
  adjDE: number;
  adjEM: number;
  adjTempo: number;
  rawOE: number;
  rawDE: number;
  gamesPlayed: number;
  lastUpdated: string;
}

// ─── Bootstrap config ─────────────────────────────────────────────────────────

export interface PriorYearData {
  teamId: string;
  teamAbbr: string;
  teamName: string;
  adjEM: number;
  adjOE: number;
  adjDE: number;
  adjTempo: number;
  finalRecord: string;      // "29-5"
  tournamentSeed?: number;
  returningMinutesPct: number;
  recruitingComposite: number;
  portalImpact: number;
  starPlayerBPM: number;
}

// ─── Tournament bracket ───────────────────────────────────────────────────────

export interface TournamentTeam {
  seed: number;
  teamId: string;
  teamAbbr: string;
  teamName: string;
  region: string;
  adjEM: number;
  eloRating: number;
  advancementProbs: Record<string, number>; // 'R32' | 'S16' | 'E8' | 'F4' | 'Champ'
  champProb: number;
}

export interface BracketSimResult {
  year: number;
  simulations: number;
  regions: Record<string, TournamentTeam[]>;
  finalFour: Array<{ team: string; prob: number }>;
  championship: Array<{ team: string; prob: number }>;
  cinderellaWatch: Array<{ team: string; seed: number; sweetSixteenProb: number }>;
}
