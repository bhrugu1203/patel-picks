# Patel Picks

A betting model for the NFL, NBA, NHL and MLB. For every player in upcoming games it projects
each stat, gives the chance of going over or under each sportsbook line, and ranks bets two ways:
the best value (expected value at the posted price) and the most likely to hit.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pandas scipy requests
.venv/bin/python -m edge.ingest --since 2024-09-01      # first time: two seasons of history (~10 min)
.venv/bin/python -m edge.backtest                       # tests the model and saves its calibration
.venv/bin/python -m edge.live                           # keeps the board current; open http://localhost:8765
```

`edge.live` refreshes lines, prices and injuries every 5 minutes, re-projects every 30 minutes
(or immediately when an injury status changes), and downloads new final scores every hour.
Change the cadence with `--every`, `--project-every` and `--books-every`.

## Hosted version

The site is published with GitHub Pages and rebuilt by GitHub Actions
(`.github/workflows/update.yml`) about every 30 minutes: it downloads new scores, grades the
pick record, re-projects every upcoming game and redeploys `web/`. The database is kept in the
Actions cache, with a daily backup on the `db` release. To use real sportsbook prices there,
add `ODDS_API_KEY` under Settings → Secrets and variables → Actions. Run an update by hand from
the Actions tab (Update board → Run workflow).

## Sportsbook prices

ESPN's free feed carries DraftKings lines and prices for NBA, NHL and MLB props and all game lines.
For NFL props it has lines but no prices. For real FanDuel, DraftKings, BetMGM, Caesars, ESPN BET,
BetRivers and Fanatics prices on every sport, get a key at https://the-odds-api.com and create `.env`:

```
ODDS_API_KEY=your_key_here
```

The free tier is 500 credits a month; one game with all NFL prop markets costs about 12 credits, so
keep `--books-every` at 30+ minutes on the free tier, or upgrade for faster refreshes. Every pull is
saved to the database, which builds a priced NFL history the backtest can use later.

## How a projection is made

1. **Volume x efficiency.** Expected minutes / targets / carries / time on ice / plate appearances,
   times the player's per-opportunity rate. Recent form is shrunk toward the player's own long-run
   rate, then toward the position average.
2. **Matchup and history** (each one is only used as much as it proved useful on past games):
   - defense vs. position: what this opponent allows to this position, relative to expectation
   - player vs. this team, and player vs. the key opposing player (NBA positional counterpart,
     NHL goalie, MLB starting pitcher)
   - opponent scheme profile from box-score tendencies (pace, three-point rate allowed, rim
     protection, run/pass funnel, pressure rate, explosive plays allowed, shot suppression,
     pitcher strikeout/walk/HR rates, lineup strikeout rate)
   - game environment from the market total and spread, home/away, back-to-backs, and usage
     freed up by injured teammates
3. **Chance of beating the line.** A negative-binomial distribution around the projection gives the
   raw over/under chance. That is then blended with the market's no-vig price using a calibration
   fitted on past games, so the model only moves off the market as far as its past disagreements
   have been right.

## Limits

- Free data has no charted scheme information (man/zone rates, coverage shells, route data);
  scheme profiles are estimated from what opponents' box scores show.
- NFL defensive matchups are team and position level; individual WR-vs-CB data is not public for free.
- MLB total bases is not available in ESPN box scores, so it is not projected.
- No model guarantees profit. The Model Report tab shows real out-of-sample results, including
  the ones that lost.
