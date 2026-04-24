// NCAAW Oracle v4.1 — SQLite Database Layer (sql.js — pure JS, no native build)
// Tables: predictions, game_results, elo_ratings, season_accuracy, weekly_accuracy, adj_em_history

import initSqlJs, { type Database as SqlJsDatabase } from 'sql.js';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import type { Prediction, GameResult, AccuracyLog, EloRating, SeasonRecord, WeekRecord } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DB_PATH = resolve(
  process.env.DB_PATH
    ? process.env.DB_PATH.startsWith('.')
      ? resolve(__dirname, '../../', process.env.DB_PATH)
      : process.env.DB_PATH
    : resolve(__dirname, '../../data/oracle.sqlite')
);

mkdirSync(dirname(DB_PATH), { recursive: true });

let _db: SqlJsDatabase | null = null;
let _SQL: Awaited<ReturnType<typeof initSqlJs>> | null = null;

// ─── Init ─────────────────────────────────────────────────────────────────────

export async function initDb(): Promise<SqlJsDatabase> {
  if (_db) return _db;

  _SQL = await initSqlJs();

  if (existsSync(DB_PATH)) {
    const buf = readFileSync(DB_PATH);
    _db = new _SQL.Database(buf);
  } else {
    _db = new _SQL.Database();
  }

  initSchema(_db);
  persistDb();
  return _db;
}

export function getDb(): SqlJsDatabase {
  if (!_db) throw new Error('DB not initialized. Call initDb() first.');
  return _db;
}

export function persistDb(): void {
  if (!_db) return;
  writeFileSync(DB_PATH, Buffer.from(_db.export()));
}

export function closeDb(): void {
  persistDb();
  _db?.close();
  _db = null;
}

// ─── Run helpers ──────────────────────────────────────────────────────────────

function run(sql: string, params: (string | number | null | undefined)[] = []): void {
  const db = getDb();
  const stmt = db.prepare(sql);
  stmt.run(params.map(p => p === undefined ? null : p));
  stmt.free();
  persistDb();
}

function queryAll<T = Record<string, unknown>>(sql: string, params: (string | number | null)[] = []): T[] {
  const db = getDb();
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const results: T[] = [];
  while (stmt.step()) results.push(stmt.getAsObject() as T);
  stmt.free();
  return results;
}

function queryOne<T = Record<string, unknown>>(sql: string, params: (string | number | null)[] = []): T | undefined {
  return queryAll<T>(sql, params)[0];
}

// ─── Schema ───────────────────────────────────────────────────────────────────

function initSchema(db: SqlJsDatabase): void {
  db.run(`
    CREATE TABLE IF NOT EXISTS predictions (
      game_id            TEXT PRIMARY KEY,
      game_date          TEXT NOT NULL,
      home_team          TEXT NOT NULL,
      away_team          TEXT NOT NULL,
      mc_prob            REAL NOT NULL,
      calibrated_prob    REAL NOT NULL,
      confidence_tier    TEXT NOT NULL,
      expected_home_score REAL,
      expected_away_score REAL,
      expected_spread    REAL,
      has_line           INTEGER DEFAULT 0,
      vegas_spread       REAL,
      vegas_ml_home      REAL,
      model_edge         REAL,
      edge_tier          TEXT,
      early_season       INTEGER DEFAULT 0,
      feature_vector     TEXT,
      model_version      TEXT,
      created_at         TEXT NOT NULL,
      actual_home_score  REAL,
      actual_away_score  REAL,
      correct            INTEGER,
      covered_spread     INTEGER
    );

    CREATE TABLE IF NOT EXISTS game_results (
      game_id     TEXT PRIMARY KEY,
      game_date   TEXT NOT NULL,
      home_team   TEXT NOT NULL,
      away_team   TEXT NOT NULL,
      home_score  REAL NOT NULL,
      away_score  REAL NOT NULL,
      home_won    INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS elo_ratings (
      team         TEXT PRIMARY KEY,
      elo          REAL NOT NULL,
      last_updated TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS season_accuracy (
      id                    INTEGER PRIMARY KEY AUTOINCREMENT,
      date                  TEXT NOT NULL,
      total_games           INTEGER NOT NULL,
      correct               INTEGER NOT NULL,
      accuracy              REAL NOT NULL,
      high_conviction_total INTEGER DEFAULT 0,
      high_conviction_correct INTEGER DEFAULT 0,
      high_conviction_accuracy REAL DEFAULT 0,
      extreme_total         INTEGER DEFAULT 0,
      extreme_correct       INTEGER DEFAULT 0,
      ats_total             INTEGER DEFAULT 0,
      ats_correct           INTEGER DEFAULT 0,
      brier_score           REAL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS weekly_accuracy (
      week        INTEGER NOT NULL,
      start_date  TEXT NOT NULL,
      total       INTEGER NOT NULL,
      correct     INTEGER NOT NULL,
      accuracy    REAL NOT NULL,
      PRIMARY KEY (week, start_date)
    );

    CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(game_date);
    CREATE INDEX IF NOT EXISTS idx_game_results_date ON game_results(game_date);
    CREATE INDEX IF NOT EXISTS idx_season_accuracy_date ON season_accuracy(date);
  `);
}

// ─── Predictions ──────────────────────────────────────────────────────────────

export function upsertPrediction(pred: Prediction): void {
  run(`
    INSERT INTO predictions (
      game_id, game_date, home_team, away_team, mc_prob, calibrated_prob,
      confidence_tier, expected_home_score, expected_away_score, expected_spread,
      has_line, vegas_spread, vegas_ml_home, model_edge, edge_tier,
      early_season, feature_vector, model_version, created_at
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT(game_id) DO UPDATE SET
      calibrated_prob = excluded.calibrated_prob,
      confidence_tier = excluded.confidence_tier,
      expected_home_score = excluded.expected_home_score,
      expected_away_score = excluded.expected_away_score,
      expected_spread = excluded.expected_spread,
      has_line = excluded.has_line,
      vegas_spread = excluded.vegas_spread,
      vegas_ml_home = excluded.vegas_ml_home,
      model_edge = excluded.model_edge,
      edge_tier = excluded.edge_tier,
      early_season = excluded.early_season,
      feature_vector = excluded.feature_vector,
      model_version = excluded.model_version
  `, [
    pred.game_id, pred.game_date, pred.home_team, pred.away_team,
    pred.mc_prob, pred.calibrated_prob, pred.confidence_tier,
    pred.expected_home_score, pred.expected_away_score, pred.expected_spread,
    pred.has_line ? 1 : 0, pred.vegas_spread ?? null, pred.vegas_ml_home ?? null,
    pred.model_edge ?? null, pred.edge_tier ?? null,
    pred.early_season ? 1 : 0,
    JSON.stringify(pred.feature_vector),
    pred.model_version, pred.created_at,
  ]);
}

export function getPredictionsByDate(date: string): Prediction[] {
  const rows = queryAll<Record<string, unknown>>(
    'SELECT * FROM predictions WHERE game_date = ? ORDER BY calibrated_prob DESC',
    [date]
  );
  return rows.map(parsePredictionRow);
}

export function updatePredictionResult(
  gameId: string,
  homeScore: number,
  awayScore: number
): void {
  const pred = queryOne<Record<string, unknown>>(
    'SELECT * FROM predictions WHERE game_id = ?', [gameId]
  );
  if (!pred) return;

  const homeWon = homeScore > awayScore;
  const predHomeWon = Number(pred['calibrated_prob']) >= 0.5;
  const correct = homeWon === predHomeWon ? 1 : 0;

  let coveredSpread: number | null = null;
  const vegasSpread = pred['vegas_spread'] as number | null;
  if (vegasSpread !== null) {
    coveredSpread = (homeScore + (vegasSpread ?? 0)) > awayScore ? 1 : 0;
  }

  run(`
    UPDATE predictions SET
      actual_home_score = ?,
      actual_away_score = ?,
      correct = ?,
      covered_spread = ?
    WHERE game_id = ?
  `, [homeScore, awayScore, correct, coveredSpread, gameId]);
}

// ─── Season accuracy ──────────────────────────────────────────────────────────

export function getSeasonRecord(season = '2025-26'): SeasonRecord {
  const [seasonStart, seasonEnd] = seasonDateRange(season);
  const rows = queryAll<Record<string, unknown>>(
    `SELECT * FROM predictions
     WHERE game_date >= ? AND game_date <= ? AND correct IS NOT NULL`,
    [seasonStart, seasonEnd]
  );

  let total = 0, correct = 0;
  let hcTotal = 0, hcCorrect = 0;
  let exTotal = 0, exCorrect = 0;
  let atsTotal = 0, atsCorrect = 0;
  let brierSum = 0, brierCount = 0;

  for (const row of rows) {
    const isCorrect = Number(row['correct']) === 1;
    const tier = String(row['confidence_tier']);
    const prob = Number(row['calibrated_prob']);
    const actualHomeWon = Number(row['actual_home_score']) > Number(row['actual_away_score']);
    const outcome = actualHomeWon ? 1 : 0;

    total++;
    if (isCorrect) correct++;

    if (tier === 'high_conviction' || tier === 'extreme') {
      hcTotal++;
      if (isCorrect) hcCorrect++;
    }
    if (tier === 'extreme') {
      exTotal++;
      if (isCorrect) exCorrect++;
    }
    if (row['covered_spread'] !== null) {
      atsTotal++;
      if (Number(row['covered_spread']) === 1) atsCorrect++;
    }

    // Brier score: (prob - outcome)^2
    brierSum += Math.pow(prob - outcome, 2);
    brierCount++;
  }

  // Weekly records
  const weekRecords = getWeekRecords(seasonStart, seasonEnd);

  return {
    total, correct,
    hc_total: hcTotal, hc_correct: hcCorrect,
    extreme_total: exTotal, extreme_correct: exCorrect,
    ats_total: atsTotal, ats_correct: atsCorrect,
    brier_sum: brierSum, games_with_brier: brierCount,
    week_records: weekRecords,
  };
}

function getWeekRecords(seasonStart: string, seasonEnd: string): WeekRecord[] {
  const rows = queryAll<Record<string, unknown>>(
    'SELECT week, start_date, total, correct, accuracy FROM weekly_accuracy WHERE start_date >= ? AND start_date <= ? ORDER BY week',
    [seasonStart, seasonEnd]
  );
  return rows.map(r => ({
    week: Number(r['week']),
    start_date: String(r['start_date']),
    total: Number(r['total']),
    correct: Number(r['correct']),
    accuracy: Number(r['accuracy']),
  }));
}

export function recordDailyAccuracy(date: string): void {
  const preds = queryAll<Record<string, unknown>>(
    'SELECT * FROM predictions WHERE game_date = ? AND correct IS NOT NULL',
    [date]
  );
  if (preds.length === 0) return;

  let total = 0, correct = 0, hcTotal = 0, hcCorrect = 0;
  let exTotal = 0, exCorrect = 0, atsTotal = 0, atsCorrect = 0;
  let brierSum = 0;

  for (const p of preds) {
    const isCorrect = Number(p['correct']) === 1;
    const tier = String(p['confidence_tier']);
    const prob = Number(p['calibrated_prob']);
    const actualWon = Number(p['actual_home_score']) > Number(p['actual_away_score']) ? 1 : 0;

    total++;
    if (isCorrect) correct++;
    if (tier === 'high_conviction' || tier === 'extreme') { hcTotal++; if (isCorrect) hcCorrect++; }
    if (tier === 'extreme') { exTotal++; if (isCorrect) exCorrect++; }
    if (p['covered_spread'] !== null) { atsTotal++; if (Number(p['covered_spread']) === 1) atsCorrect++; }
    brierSum += Math.pow(prob - actualWon, 2);
  }

  const acc = total > 0 ? correct / total : 0;
  const hcAcc = hcTotal > 0 ? hcCorrect / hcTotal : 0;
  const brier = total > 0 ? brierSum / total : 0;

  run(`
    INSERT INTO season_accuracy (date, total_games, correct, accuracy,
      high_conviction_total, high_conviction_correct, high_conviction_accuracy,
      extreme_total, extreme_correct, ats_total, ats_correct, brier_score)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT DO NOTHING
  `, [date, total, correct, acc, hcTotal, hcCorrect, hcAcc, exTotal, exCorrect, atsTotal, atsCorrect, brier]);
}

export function getRecentAccuracy(days = 7): { total: number; correct: number; accuracy: number } {
  const since = new Date(Date.now() - days * 86400000).toISOString().split('T')[0];
  const rows = queryAll<Record<string, unknown>>(
    'SELECT SUM(total_games) as t, SUM(correct) as c FROM season_accuracy WHERE date >= ?',
    [since]
  );
  const total = Number(rows[0]?.['t'] ?? 0);
  const correct = Number(rows[0]?.['c'] ?? 0);
  return { total, correct, accuracy: total > 0 ? correct / total : 0 };
}

// ─── Game results ─────────────────────────────────────────────────────────────

export function upsertGameResult(result: GameResult): void {
  run(`
    INSERT OR REPLACE INTO game_results (game_id, game_date, home_team, away_team, home_score, away_score, home_won)
    VALUES (?,?,?,?,?,?,?)
  `, [result.game_id, result.game_date, result.home_team, result.away_team,
     result.home_score, result.away_score, result.home_won ? 1 : 0]);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function parsePredictionRow(row: Record<string, unknown>): Prediction {
  return {
    game_id: String(row['game_id']),
    game_date: String(row['game_date']),
    home_team: String(row['home_team']),
    away_team: String(row['away_team']),
    mc_prob: Number(row['mc_prob']),
    calibrated_prob: Number(row['calibrated_prob']),
    confidence_tier: String(row['confidence_tier']) as Prediction['confidence_tier'],
    expected_home_score: Number(row['expected_home_score']),
    expected_away_score: Number(row['expected_away_score']),
    expected_spread: Number(row['expected_spread']),
    has_line: Number(row['has_line']) === 1,
    vegas_spread: row['vegas_spread'] != null ? Number(row['vegas_spread']) : undefined,
    vegas_ml_home: row['vegas_ml_home'] != null ? Number(row['vegas_ml_home']) : undefined,
    model_edge: row['model_edge'] != null ? Number(row['model_edge']) : undefined,
    edge_tier: row['edge_tier'] != null ? String(row['edge_tier']) as Prediction['edge_tier'] : undefined,
    early_season: Number(row['early_season']) === 1,
    feature_vector: row['feature_vector'] ? JSON.parse(String(row['feature_vector'])) : {},
    model_version: String(row['model_version'] ?? '4.1'),
    created_at: String(row['created_at']),
    actual_home_score: row['actual_home_score'] != null ? Number(row['actual_home_score']) : undefined,
    actual_away_score: row['actual_away_score'] != null ? Number(row['actual_away_score']) : undefined,
    correct: row['correct'] != null ? Number(row['correct']) === 1 : undefined,
    covered_spread: row['covered_spread'] != null ? Number(row['covered_spread']) === 1 : undefined,
  };
}

function seasonDateRange(season: string): [string, string] {
  // '2025-26' → Nov 2025 – Apr 2026
  const year = parseInt(season.split('-')[0]);
  return [`${year}-11-01`, `${year + 1}-04-10`];
}
