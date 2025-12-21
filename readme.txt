My Chess Notebook — White
==========================

Overview
--------
Streamlit app for building a personal White-side chess repertoire. Enter lines in SAN, attach comments per ply, auto-import comments from matching saved prefixes, explore or edit saved games, and train tactics sliced from your own lines.

Key Features
------------
- Move entry with lenient SAN parsing, next-move suggestions learned from all saved lines.
- Comments attached to plies; imported comments from other games with the same prefix; walkthrough mode hides text until revealed.
- Explore vs. Edit modes plus “Use as Template” to branch a saved line without overwriting originals.
- Tactics: slice a saved line (White to move) into a training sequence; randomize practice with spoilers (tags, solution, notes).
- Coverage map showing how many saved lines reach each position.
- Search saved games by a specific move and side; view similar games sharing the longest prefix.

Setup
-----
1) Ensure Python 3.11+ is available.  
2) (Recommended) Create a virtual env in the repo: `python3 -m venv .venv` and activate it.  
3) Install dependencies: `pip install -r requirements.txt`.

Run the App
-----------
From the project root:
    streamlit run app.py

Data & Files
------------
- Saved games: `data/games/game_<timestamp>.json` (SAN moves, source, URL, comments).
- Saved tactics: `data/tactics/tactic_<timestamp>_<id>.json`.
- Directories are created on first run; files are plain JSON for easy backup or sync.

Typical Workflow
----------------
- New line: enter moves (suggestions appear when a prefix exists), add comments at the last played ply, and save with a Source name.
- Open saved line: loads in Explore mode (moves locked, comments editable in their source game); switch to Edit to change moves and overwrite.
- Template: “Use as Template” clones the move list as a new unsaved line with comments imported from the original.
- Walkthrough: step through a line with comments hidden until revealed under the board.
- Tactics: in a saved game, pick a start (after a Black move) and end (after a White move) window, add notes, and save. Train via “Tactic mode” with optional tag filters.
- Coverage: adjust the move-depth slider to see how many saved lines reach each node in the repertoire tree.

Shortcuts & Tips
----------------
- Move input accepts O-O, 0-0, o-o (and queenside variants) plus lenient capture/check notation.
- Undo only at the line end; navigation buttons jump through the move list.
- “Update all games” re-syncs imported comments from their originals.
