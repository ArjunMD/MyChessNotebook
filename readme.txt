My Chess Notebook — White
==========================


Overview
--------
Streamlit app for building a personal White-side chess repertoire.

Key Features
------------
- New game: Build and save your own line
- Move suggestions based on history: Building new lines gets easier as you save more
- Attach comments to moves
- Comment propogation: Comments will propgate to all games that match to that spot. Comments can only be edited from the original game.
- Search saved lines by specific moves
- Walkthrough mode: See propogated comments in realtime as you study a new or existing line
- Explore mode: Moves are read-only, comments can still be edited 
- Game info and save: Save name of game and URL if there is one
- Matching games: When building a new line, see which games are matching so far
- Similar Games: When reviewing an old line, see games that were matched the longest
- Coverage map: Global view of saved lines
- Import/Export: Import from PGN or a friend's notebook; export your notebook for backup or sharing
- Tactics: Slice a saved line into a tactic. Auto-tag and write notes
- Tactic mode: Practice saved tactics, filter which tactics appear by relevant pieces 

Setup
-----
- Ensure Python 3.11+ is available.  
- (Recommended) Create a virtual env in the repo: `python3 -m venv .venv` and activate it.  
- Install dependencies: `pip install -r requirements.txt`.
- From the project root: streamlit run app.py

Data & Files
------------
- Saved games: `data/games/game_<timestamp>.json` (Moves, source, URL, comments).
- Saved tactics: `data/tactics/tactic_<timestamp>_<id>.json`.
- Directories are created on first run; files are plain JSON for easy backup or sync.

Typical Workflow
----------------
- New line: enter moves (suggestions appear when a prefix exists), add comments, and save.
- Open saved line: loads in Explore mode (moves locked, comments editable in their source game); switch to Edit to change moves and overwrite.
- Template: “Use as Template” clones the move list as a new unsaved line with comments imported from the original.
- Walkthrough: step through a line with comments hidden until revealed under the board.
- Tactics: in a saved game, pick a start (after a Black move) and end (after a White move) window, add notes, and save. Train via “Tactic mode” with optional tag filters.
- Coverage: adjust the move-depth slider to see how many saved lines reach each node in the repertoire tree.

Tips
----------------
- Move input accepts O-O, 0-0, o-o (and queenside variants) plus lenient capture/check notation.
- “Update all games” re-syncs imported comments from their originals. Useful after imports or significant edits. 
