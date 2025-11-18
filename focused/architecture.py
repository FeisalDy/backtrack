"""
Architecture diagram generator.
Creates a visual representation of the module structure.
"""

architecture_diagram = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        BACKTEST SYSTEM ARCHITECTURE                           ║
║                     Single Responsibility Principle (SRP)                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│                                  main.py                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Orchestrator - Entry Point & Workflow Coordination                    │  │
│  │  • Coordinates all modules                                             │  │
│  │  • Manages parallel execution                                          │  │
│  │  • Displays results                                                    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────┬──────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┬──────────────────┐
        │               │               │                  │
        ▼               ▼               ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌────────────────┐ ┌──────────────────┐
│  config.py   │ │ data_loader  │ │ backtest_engine│ │  visualization   │
│              │ │     .py      │ │     .py        │ │      .py         │
│ Configuration│ │ Data Loading │ │ Backtest Exec  │ │ Chart Generation │
│              │ │              │ │                │ │                  │
│ • Symbols    │ │ • Yahoo Fin  │ │ • Main loop    │ │ • Equity curve   │
│ • Capital    │ │ • OHLCV data │ │ • Position size│ │ • P&L charts     │
│ • Fees       │ │ • Memory opt │ │ • Metrics calc │ │ • Comparisons    │
│ • Thresholds │ │              │ │                │ │                  │
└──────────────┘ └──────┬───────┘ └───────┬────────┘ └──────────────────┘
                        │                 │
                        │                 │
                        ▼                 │
                 ┌──────────────┐         │
                 │ indicators   │         │
                 │     .py      │         │
                 │ Indicators   │         │
                 │              │         │
                 │ • MA         │         │
                 │ • ATR        │         │
                 │ • OBV        │◄────────┤
                 │ • Momentum   │         │
                 │ • Z-scores   │         │
                 └──────┬───────┘         │
                        │                 │
                        ▼                 │
                 ┌──────────────┐         │
                 │ bias_scoring │         │
                 │     .py      │         │
                 │ Bias Calc    │         │
                 │              │         │
                 │ • Vectorized │         │
                 │ • 7 factors  │◄────────┤
                 │ • Bull/Bear  │         │
                 │ • Thresholds │         │
                 └──────────────┘         │
                                          │
                        ┌─────────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  trading.py  │
                 │              │
                 │ Trade Logic  │
                 │              │
                 │ • TP/SL calc │
                 │ • Stop check │
                 │ • Vectorized │
                 │ • Risk-Reward│
                 └──────────────┘

═══════════════════════════════════════════════════════════════════════════════

DATA FLOW:

1. main.py          → Loads config, spawns parallel workers
                    ↓
2. backtest_engine  → Receives (symbol, interval, limit)
                    ↓
3. data_loader      → Fetches OHLCV from Yahoo Finance
                    ↓
4. indicators       → Calculates MA, ATR, OBV, momentum, etc.
                    ↓
5. bias_scoring     → Computes bull/bear bias for each bar
                    ↓
6. trading          → Calculates TP/SL, checks stops (vectorized)
                    ↓
7. backtest_engine  → Executes trades, calculates P&L
                    ↓
8. visualization    → Generates charts and saves results
                    ↓
9. main.py          → Aggregates results, creates comparison

═══════════════════════════════════════════════════════════════════════════════

MODULE SIZES (Lines of Code):

config.py           →   ~50 lines   (Parameters only)
data_loader.py      →   ~60 lines   (Single function + docs)
indicators.py       →   ~80 lines   (Pure calculation)
bias_scoring.py     →  ~150 lines   (Complex scoring logic)
trading.py          →   ~80 lines   (Level calc + stop check)
backtest_engine.py  →  ~180 lines   (Main backtest loop)
visualization.py    →  ~200 lines   (3 plotting functions)
main.py             →   ~60 lines   (Orchestration only)
─────────────────────────────────────────────────────────────────────────────
TOTAL:              →  ~860 lines   vs. 700 lines (monolithic)

Slight increase in LOC due to:
- Module docstrings
- Clear separation and imports
- Better documentation

But MASSIVE improvement in:
✅ Maintainability    ✅ Testability      ✅ Reusability
✅ Readability        ✅ Scalability      ✅ Modularity

═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(architecture_diagram)
