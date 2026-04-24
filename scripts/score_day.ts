// Score a past day's predictions against actual results
import 'dotenv/config';
import { initDb, closeDb, getPredictionsByDate, updatePredictionResult, recordDailyAccuracy } from '../src/db/database.js';
import { fetchCompletedGames } from '../src/api/espnClient.js';

const dateStr = process.argv[2] ?? '2026-01-15';

await initDb();
const results = await fetchCompletedGames(dateStr);
console.log(`Completed games on ${dateStr}: ${results.length}`);

for (const r of results) {
  updatePredictionResult(r.gameId, r.homeScore, r.awayScore);
}
recordDailyAccuracy(dateStr);

const preds = getPredictionsByDate(dateStr).filter(p => p.correct !== undefined);
console.log(`Predictions with results: ${preds.length}`);
preds.forEach(p => console.log(
  `${p.away_team} @ ${p.home_team} | ${p.correct ? 'CORRECT' : 'WRONG'} | ${p.actual_home_score}-${p.actual_away_score}`
));

closeDb();
