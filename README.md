# Capstone Project

## Can a Defense Predict Run vs. Pass Before the Snap? Final Report

**Notebooks**
- [01_data_cleaning_eda_final.ipynb](01_data_cleaning_eda_final.ipynb): cleans the 2025 data and explores it
- [02_league_model_final.ipynb](02_league_model_final.ipynb): trains, tunes, and tests league-wide models
- [03_team_scouting_tool_final.ipynb](03_team_scouting_tool_final.ipynb): builds a call sheet for one opponent
- [call_sheet_LA_week4_final.html](call_sheet_LA_week4_final.html): sample call sheet (Rams, film from weeks 1-3)

### How to Use

The notebooks are built for **Google Colab**. Run them in order with **Runtime → Run all**.

| Notebook | What you do | What it exports |
|---|---|---|
| **01 Data cleaning** | Run all. It downloads the 2025 data with `nflreadpy` (needs internet) | `pbp_2025_clean.csv`, downloaded through your browser. Upload it to the **📁 Files** panel for Notebooks 2 and 3 |
| **02 League model** | Upload `pbp_2025_clean.csv` to Colab's **📁 Files** panel, then run all | Nothing. All results print in the notebook (Section 19 downloads formation data, so it needs internet) |
| **03 Scouting tool** | Upload `pbp_2025_clean.csv` to the **📁 Files** panel. In Section 2, set `TEAM` (e.g., `"SEA"`) and `WEEK_RANGE` (e.g., `"1-3"`), or set them to `None` to type them in. Then run all | `call_sheet_<TEAM>_week<N>.html`, downloaded through your browser. Open it in a browser to read or print it |

⚠️ Files uploaded to the **📁 Files** panel are deleted when the Colab session ends, so upload the CSV again each time you open Notebook 2 or 3.

**Files provided to help:**
- **`pbp_2025_clean.csv`**: the cleaned data from Notebook 1, so you can go straight to Notebooks 2 and 3 without running Notebook 1.
- **`call_sheet_LA_week4_final.html`**: a finished call sheet, so you can see what Notebook 3 produces before running it.

### 1. Problem Statement

A defensive coordinator has to make a call before the ball is snapped. If the defense can guess run or pass from what it sees before the snap (down, distance, score, clock, and the players who run onto the field), it can be in the right call more often.

**Goal:** predict run vs. pass for the whole league, and build a scouting tool that shows one opponent's habits before a game.

### 2. Model Outcomes or Predictions

- **Type of learning:** supervised **classification** (two classes: run or pass).
- **Target:** `qb_dropback` (1 = pass, including sacks and scrambles; 0 = designed run). It measures the play that was *called*, not how it ended.
- **Output:** a predicted call and a pass probability for each play.
- **Benchmark:** nflfastR's expected-pass model (`xpass`), used only for comparison and never as a feature.

### 3. Data Acquisition

Two 2025 datasets from `nflreadpy`, joined on `game_id` + `play_id`:

| Source | What it adds |
|---|---|
| `load_pbp()` | Down, distance, field position, score, clock, timeouts, home/away, no-huddle, `xpass` |
| `load_participation()` | Offensive personnel for each play (e.g., 1 RB, 1 TE, 3 WR) |

Down & distance shows that the data can separate runs from passes. 1st down is a coin flip, while 3rd & 4+ is over 90% pass:

![Run/pass % by down & distance](images/run_pass_by_down_distance.png)

### 4. Data Preprocessing/Preparation

**a. Missing values and inconsistencies**
- Kept only run and pass plays (removed punts, kicks, kneels, spikes, and penalties).
- Removed two-point conversions, which have no down. This also removed the only missing `xpass` values.
- Parsed personnel text into counts: `n_rb` (RB + FB), `n_te`, `n_wr`, `n_ol`.
- Dropped 16 plays where the players didn't add up to 10 or 11 (charting errors).
- Final data: **34,486 plays**, no missing values, no duplicates.

**b. Train/test split**
- Split **by week** so no game is in both sets: train on weeks 1-13, test on weeks 14-17.
- Weeks 14-17 give every team about 4 test games and leave out week 18 and the playoffs, which are checked separately.

**c. Analysis and encoding**
- `is_home` turns home/away into 0/1. Down & distance buckets and personnel groups were built for charts and baselines.
- `qtr` was left out because it overlaps with `game_seconds_remaining` (r = -0.96).
- `n_rb` was left out so personnel effects read as swaps (for example, "a WR in for a RB").
- Numbers were scaled inside a pipeline, so the test weeks never affected the scaling.

### 5. Modeling

Three classification models, each tuned with **GridSearchCV** using **GroupKFold by week**:

| Model | Setting tuned | Best |
|---|---|---|
| Logistic Regression | `C` | 1 |
| Decision Tree | `max_depth`, `min_samples_leaf` | 8, 200 |
| KNN | `n_neighbors` | 151 |

Each model was compared with simple baselines (always guess pass; majority call by down & distance or personnel) and with xpass.

### 6. Model Evaluation

**Metrics:** **accuracy** (easy to explain to a coach and comparable to xpass), **runs caught** (missing a run hurts a defense most), and **log loss** (are the pass percentages trustworthy?).

| Model (test weeks 14-17, 7,470 plays) | Accuracy | Runs caught | Log loss |
|---|---|---|---|
| Always guess pass | 0.591 | 0% | 0.677 |
| **xpass (benchmark)** | **0.711** | **62%** | **0.541** |
| Logistic Regression | 0.692 | 54% | 0.575 |
| Decision Tree | 0.693 | 52% | 0.559 |
| KNN | 0.682 | 47% | 0.578 |

**Findings**
- All three models beat guessing by about **10 points**, but **xpass is about 2 points better** (margin of error ±1 point).
- **Logistic Regression is the best choice.** It ties the tree on accuracy, and its coefficients explain *why*, which a coach can use.
- **Down and distance matter most,** then personnel. Swapping a RB for a WR almost doubles the odds of a pass, and a 6th lineman is a strong run sign.
- **Catching more runs:** raising the pass cutoff to 0.63 lets the model catch 78% of runs instead of 54%, at a cost of about 1 point of accuracy.
- **The results hold in week 18 and the playoffs** (every model within half a point).
- **Formation would help but comes too late.** Adding shotgun raises accuracy to 73.5%, but a defense usually can't see it before it calls the play.

**Scouting tool (Notebook 3):** a coach picks an opponent and the weeks of film they have, and gets a printable call sheet comparing that team to the league. Example: in weeks 1-3, the Rams passed only 35% of the time with 2 TEs (league: 49%), and the odds of a pass dropped to less than one-third when a TE came in for a WR.

**Next steps:** add more seasons, let the coach choose the run/pass cutoff, and add a team adjustment on top of xpass.
