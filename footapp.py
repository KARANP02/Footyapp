import streamlit as st
import pandas as pd
import random
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import libsql_client

st.set_page_config(
    page_title="Football Team Manager",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FootballDatabase:
    def __init__(self):
        pass
    
    def get_connection(self):
        url = st.secrets["TURSO_DATABASE_URL"]
        auth_token = st.secrets["TURSO_AUTH_TOKEN"]
        
        # Use HTTPS for free Turso plan (no WebSocket needed)
        if url.startswith("libsql://"):
            url = url.replace("libsql://", "https://")
        
        return libsql_client.create_client_sync(url=url, auth_token=auth_token)

    def init_database(self):
        conn = self.get_connection()
        try:
            # --- CHANGE: Added 'games_drawn' column to the players table ---
            conn.execute('''
                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, 
                    is_active INTEGER DEFAULT 1, position1 TEXT NOT NULL, 
                    position2 TEXT, position3 TEXT, technical_skill REAL NOT NULL, 
                    physical_strength REAL NOT NULL, stamina REAL NOT NULL, 
                    speed REAL NOT NULL, shot_accuracy REAL NOT NULL, 
                    experience REAL NOT NULL, goals INTEGER DEFAULT 0, 
                    saves INTEGER DEFAULT 0, assists INTEGER DEFAULT 0,
                    games_won INTEGER DEFAULT 0, games_lost INTEGER DEFAULT 0,
                    games_drawn INTEGER DEFAULT 0, 
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS games (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT NOT NULL,
                    team_a_score INTEGER NOT NULL, team_b_score INTEGER NOT NULL,
                    team_a_players TEXT, team_b_players TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS game_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL,
                    player_name TEXT NOT NULL, goals INTEGER DEFAULT 0,
                    assists INTEGER DEFAULT 0, saves INTEGER DEFAULT 0,
                    FOREIGN KEY (game_id) REFERENCES games (id)
                )
            ''')
        finally:
            conn.close()

    def add_player(self, conn, name: str, position1: str, position2: str, position3: str, 
                   technical_skill: float, physical_strength: float, stamina: float, 
                   speed: float, shot_accuracy: float, experience: float):
        try:
            conn.execute('''
            INSERT INTO players (name, position1, position2, position3, technical_skill, physical_strength, stamina, speed, shot_accuracy, experience)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (name, position1, position2, position3, technical_skill, physical_strength, stamina, speed, shot_accuracy, experience))
            return True
        except:
            return False
    
    def get_all_players(self):
        conn = self.get_connection()
        try:
            result = conn.execute("SELECT * FROM players WHERE is_active = 1 ORDER BY name")
            if result.rows:
                return pd.DataFrame(result.rows, columns=result.columns)
            return pd.DataFrame()
        finally:
            conn.close()
            
    def _update_player_stats(self, tx, player_name: str, stat_type: str, increment: int = 1):
        tx.execute(f'UPDATE players SET {stat_type} = {stat_type} + ? WHERE name = ?', (increment, player_name))

    def save_game(self, game_date, team_a_score: int, team_b_score: int, 
                    team_a_players: list, team_b_players: list, 
                    goal_counts: dict, assist_counts: dict, save_counts: dict):
        conn = self.get_connection()
        try:
            rs = conn.execute(
                "INSERT INTO games (date, team_a_score, team_b_score, team_a_players, team_b_players) VALUES (?, ?, ?, ?, ?)",
                [game_date.strftime("%Y-%m-%d"), team_a_score, team_b_score, 
                ','.join(team_a_players), ','.join(team_b_players)]
            )
            game_id = rs.last_insert_rowid
            
            statements = []
            
            # --- CHANGE: Added logic to handle draws ---
            is_draw = team_a_score == team_b_score
            
            if is_draw:
                for player in team_a_players + team_b_players:
                    statements.append(("UPDATE players SET games_drawn = games_drawn + 1 WHERE name = ?", [player]))
            else:
                team_a_won = team_a_score > team_b_score
                for player in team_a_players:
                    if team_a_won:
                        statements.append(("UPDATE players SET games_won = games_won + 1 WHERE name = ?", [player]))
                    else:
                        statements.append(("UPDATE players SET games_lost = games_lost + 1 WHERE name = ?", [player]))
                
                for player in team_b_players:
                    if not team_a_won:
                        statements.append(("UPDATE players SET games_won = games_won + 1 WHERE name = ?", [player]))
                    else:
                        statements.append(("UPDATE players SET games_lost = games_lost + 1 WHERE name = ?", [player]))
            
            for player, count in goal_counts.items():
                if count > 0: statements.append(("UPDATE players SET goals = goals + ? WHERE name = ?", [count, player]))
            
            for player, count in assist_counts.items():
                if count > 0: statements.append(("UPDATE players SET assists = assists + ? WHERE name = ?", [count, player]))
            
            for player, count in save_counts.items():
                if count > 0: statements.append(("UPDATE players SET saves = saves + ? WHERE name = ?", [count, player]))
            
            all_game_stats = {}
            for p, g in goal_counts.items(): all_game_stats.setdefault(p, {})['goals'] = g
            for p, a in assist_counts.items(): all_game_stats.setdefault(p, {})['assists'] = a
            for p, s in save_counts.items(): all_game_stats.setdefault(p, {})['saves'] = s
            
            for player_name, stats in all_game_stats.items():
                if any(v > 0 for v in stats.values()):
                    statements.append((
                        "INSERT INTO game_stats (game_id, player_name, goals, assists, saves) VALUES (?, ?, ?, ?, ?)",
                        [game_id, player_name, stats.get('goals', 0), stats.get('assists', 0), stats.get('saves', 0)]
                    ))
            
            if statements:
                conn.batch(statements)
                
        finally:
            conn.close()

    def delete_game(self, game_id: int):
        conn = self.get_connection()
        try:
            rs = conn.execute("SELECT * FROM games WHERE id = ?", [game_id])
            if not rs.rows: 
                return
            game_data = dict(zip(rs.columns, rs.rows[0]))
            
            stats_rs = conn.execute("SELECT player_name, goals, assists, saves FROM game_stats WHERE game_id = ?", [game_id])
            
            statements = []
            
            for row in stats_rs.rows:
                player_name, goals, assists, saves = row
                if goals > 0: statements.append(("UPDATE players SET goals = goals - ? WHERE name = ?", [goals, player_name]))
                if assists > 0: statements.append(("UPDATE players SET assists = assists - ? WHERE name = ?", [assists, player_name]))
                if saves > 0: statements.append(("UPDATE players SET saves = saves - ? WHERE name = ?", [saves, player_name]))
            
            # --- CHANGE: Added logic to revert draws ---
            is_draw = game_data['team_a_score'] == game_data['team_b_score']
            team_a_players = game_data['team_a_players'].split(',')
            team_b_players = game_data['team_b_players'].split(',')

            if is_draw:
                for player in team_a_players + team_b_players:
                    statements.append(("UPDATE players SET games_drawn = games_drawn - 1 WHERE name = ?", [player]))
            else:
                team_a_won = game_data['team_a_score'] > game_data['team_b_score']
                for player in team_a_players:
                    if team_a_won:
                        statements.append(("UPDATE players SET games_won = games_won - 1 WHERE name = ?", [player]))
                    else:
                        statements.append(("UPDATE players SET games_lost = games_lost - 1 WHERE name = ?", [player]))
                
                for player in team_b_players:
                    if not team_a_won:
                        statements.append(("UPDATE players SET games_won = games_won - 1 WHERE name = ?", [player]))
                    else:
                        statements.append(("UPDATE players SET games_lost = games_lost - 1 WHERE name = ?", [player]))
            
            statements.append(("DELETE FROM game_stats WHERE game_id = ?", [game_id]))
            statements.append(("DELETE FROM games WHERE id = ?", [game_id]))
            
            if statements:
                conn.batch(statements)
            
        finally:
            conn.close()
    
    def get_games_history(self):
        conn = self.get_connection()
        try:
            result = conn.execute("SELECT * FROM games ORDER BY date DESC")
            if result.rows:
                return pd.DataFrame(result.rows, columns=result.columns)
            return pd.DataFrame()
        finally:
            conn.close()
            
    def get_player_game_history(self, player_name: str, limit: int = 5):
        conn = self.get_connection()
        try:
            query = """
            SELECT g.date, g.team_a_players, g.team_b_players, g.team_a_score, g.team_b_score,
                   COALESCE(gs.goals, 0) as goals, COALESCE(gs.assists, 0) as assists, COALESCE(gs.saves, 0) as saves
            FROM games g
            LEFT JOIN game_stats gs ON g.id = gs.game_id AND gs.player_name = ?
            WHERE g.team_a_players LIKE ? OR g.team_b_players LIKE ?
            ORDER BY g.date DESC, g.id DESC LIMIT ?
            """
            result = conn.execute(query, [player_name, f'%{player_name}%', f'%{player_name}%', limit])
            if result.rows:
                return pd.DataFrame(result.rows, columns=result.columns)
            return pd.DataFrame()
        finally:
            conn.close()

    def deactivate_player(self, player_name: str):
        conn = self.get_connection()
        try:
            conn.execute("UPDATE players SET is_active = 0 WHERE name = ?", (player_name,))
        finally:
            conn.close()

# TeamBalancer class remains the same
class TeamBalancer:
    def __init__(self): self.position_weights = {'DEF': {'priority': 1, 'max_per_team': 4},'MID': {'priority': 2, 'max_per_team': 4},'FWD': {'priority': 3, 'max_per_team': 3}}
    def _calculate_overall_rating(self, player: Dict) -> float:
        position = player['position1']
        if position == 'DEF': return (player['physical_strength'] * 0.3 + player['technical_skill'] * 0.25 + player['stamina'] * 0.2 + player['experience'] * 0.15 + player['speed'] * 0.05 + player['shot_accuracy'] * 0.05)
        elif position == 'MID': return (player['technical_skill'] * 0.35 + player['stamina'] * 0.25 + player['speed'] * 0.15 + player['experience'] * 0.15 + player['physical_strength'] * 0.05 + player['shot_accuracy'] * 0.05)
        elif position == 'FWD': return (player['shot_accuracy'] * 0.35 + player['speed'] * 0.25 + player['technical_skill'] * 0.15 + player['experience'] * 0.15 + player['stamina'] * 0.05 + player['physical_strength'] * 0.05)
        else: return (player['technical_skill'] + player['physical_strength'] + player['stamina'] + player['speed'] + player['shot_accuracy'] + player['experience']) / 6
    def generate_balanced_teams(self, players_df: pd.DataFrame) -> Dict:
        if len(players_df) < 2: return None
        players_list = players_df.to_dict('records')
        best_teams = self._find_best_combination(players_list)
        return {'team_a': best_teams['team_a'], 'team_b': best_teams['team_b'],'balance_score': self._calculate_balance_score(best_teams['team_a'], best_teams['team_b']),'analysis': self._get_team_analysis(best_teams['team_a'], best_teams['team_b'])}
    def _find_best_combination(self, players: List[Dict]) -> Dict:
        best_combination, best_score = None, float('inf')
        for _ in range(100):
            random.shuffle(players)
            team_a, team_b = players[:len(players)//2], players[len(players)//2:]
            score = abs(self._get_team_overall_average(team_a) - self._get_team_overall_average(team_b))
            if score < best_score:
                best_score, best_combination = score, {'team_a': team_a, 'team_b': team_b}
        return best_combination
    def _get_team_overall_average(self, team: List[Dict]) -> float:
        if not team: return 0
        return sum(self._calculate_overall_rating(p) for p in team) / len(team)
    def _calculate_balance_score(self, team_a: List[Dict], team_b: List[Dict]) -> int:
        imbalance = abs(self._get_team_overall_average(team_a) - self._get_team_overall_average(team_b))
        return max(0, min(100, int(100 - imbalance * 20)))
    def _get_team_analysis(self, team_a: List[Dict], team_b: List[Dict]) -> Dict:
        avg_a, avg_b = self._get_team_overall_average(team_a), self._get_team_overall_average(team_b)
        return {'team_a_avg_skill': round(avg_a, 1), 'team_b_avg_skill': round(avg_b, 1), 'skill_difference': round(abs(avg_a - avg_b), 2)}

# --- UI PAGES ---
def create_game_page(db, balancer):
    st.header("Create New Game")
    players_df = db.get_all_players()
    if players_df.empty:
        st.warning("⚠️ No players found! Please add or import players in the 'Manage Players' section.")
        return
    st.subheader("Select Available Players")
    cols = st.columns(4)
    selected_players = []
    for i, (_, player) in enumerate(players_df.iterrows()):
        col_idx = i % 4
        with cols[col_idx]:
            positions = [p for p in [player['position1'], player['position2'], player['position3']] if p]
            position_str = '/'.join(positions)
            if st.checkbox(f"{player['name']} ({position_str})", key=f"player_{player['id']}"):
                selected_players.append(player)
    if len(selected_players) >= 2:
        if st.button("🎯 Generate Balanced Teams", type="primary"):
            st.session_state['generated_teams'] = balancer.generate_balanced_teams(pd.DataFrame(selected_players))
            st.rerun()
    else:
        st.info("👆 Please select at least 2 players to generate teams")
    if 'generated_teams' in st.session_state and st.session_state.get('generated_teams'):
        teams = st.session_state['generated_teams']
        st.subheader("Generated Teams")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔴 Team A")
            st.dataframe(pd.DataFrame(teams['team_a'])[['name', 'position1']], hide_index=True)
        with col2:
            st.markdown("### 🔵 Team B")
            st.dataframe(pd.DataFrame(teams['team_b'])[['name', 'position1']], hide_index=True)
        
        st.subheader("Record Game Results")
        game_date = st.date_input("Game Date", value=datetime.now())
        c1, c2 = st.columns(2)
        team_a_score = c1.number_input("Team A Score", min_value=0, value=0, step=1)
        team_b_score = c2.number_input("Team B Score", min_value=0, value=0, step=1)
        st.markdown("---")
        st.subheader("Enter Detailed Stats")
        all_players_in_game = [p['name'] for p in teams['team_a'] + teams['team_b']]
        stats_df = pd.DataFrame({"Player": sorted(all_players_in_game), "Goals": 0, "Assists": 0, "Saves": 0})
        edited_stats_df = st.data_editor(stats_df, hide_index=True, key="stats_editor")
        if st.button("💾 Save Game Results", type="primary"):
            team_a_players, team_b_players = [p['name'] for p in teams['team_a']], [p['name'] for p in teams['team_b']]
            goal_counts = {row["Player"]: row["Goals"] for _, row in edited_stats_df.iterrows()}
            assist_counts = {row["Player"]: row["Assists"] for _, row in edited_stats_df.iterrows()}
            save_counts = {row["Player"]: row["Saves"] for _, row in edited_stats_df.iterrows()}
            db.save_game(game_date, team_a_score, team_b_score, team_a_players, team_b_players, goal_counts, assist_counts, save_counts)
            st.success("🎉 Game saved successfully!")
            del st.session_state['generated_teams']
            st.rerun()

def manage_players_page(db):
    st.header("Manage Players")

    # --- CHANGE: Wrapped the bulk import expander in an admin check ---
    if st.session_state.get('is_admin'):
        with st.expander("🚀 Bulk Import Players", expanded=False):
            st.info("This adds the default list of players. Players that already exist will be skipped.")
            if st.button("📥 Import All Players", type="primary"):
                players_data = [
                    ("Shivam", "FWD", "", "", 10, 6.33, 9, 9.67, 10, 10), ("Jai", "FWD", "MID", "", 6.33, 6.67, 7.33, 7, 7, 6.33),
                    ("Nim", "DEF", "MID", "", 7.17, 8, 6.83, 7, 7.67, 8.33), ("Leigh", "DEF", "FWD", "", 6.5, 8.33, 5.67, 6.33, 5.67, 5.83),
                    ("Rohan", "FWD", "", "", 8.33, 6, 9.67, 9.33, 8.67, 8), ("Shish", "FWD", "MID", "DEF", 9.17, 7.83, 9.33, 8.67, 7, 10),
                    ("Ama", "FWD", "DEF", "", 7.33, 10, 6.33, 6.67, 7.5, 7.67), ("Amarjot", "DEF", "", "", 7.67, 8.83, 6.67, 6.33, 8.83, 9),
                    ("Muj", "FWD", "DEF", "", 7, 7.17, 5.33, 5, 6.17, 7.17), ("Rahul", "DEF", "", "", 5.83, 9.67, 7.5, 6.67, 2.67, 4.33),
                    ("Jigs", "FWD", "", "", 5.33, 4.67, 6.5, 6.5, 6.17, 6), ("Vin", "MID", "", "", 3.33, 8.83, 6, 6, 1.67, 3.33),
                    ("Mit", "MID", "", "", 6.67, 6.33, 5.83, 5.83, 6.67, 6.83), ("Lustin", "MID", "", "", 4.33, 5.33, 5, 4.5, 4, 4.67),
                    ("Aj", "DEF", "MID", "", 6.83, 7.33, 7, 6.5, 6.5, 7.67), ("Karan", "DEF", "MID", "", 5.67, 7, 6, 6.5, 5.67, 5),
                    ("Lach", "DEF", "MID", "", 4, 6, 5, 5, 4, 4)
                ]
                conn = db.get_connection()
                try:
                    success_count, error_count = 0, 0
                    progress_bar = st.progress(0, text="Starting import...")
                    for i, data in enumerate(players_data):
                        progress_bar.progress((i + 1) / len(players_data), text=f"Importing {data[0]}...")
                        if db.add_player(conn, *data): success_count += 1
                        else: error_count += 1
                    progress_bar.empty()
                    if success_count > 0: st.success(f"✅ Added {success_count} new players!")
                    if error_count > 0: st.warning(f"⚠️ {error_count} players were skipped (already exist).")
                finally:
                    conn.close()
                st.rerun()

    with st.expander("➕ Add New Player", expanded=True):
        with st.form("new_player_form"):
            name = st.text_input("Player Name")
            c1, c2, c3 = st.columns(3)
            position1 = c1.selectbox("Position 1", ["DEF", "MID", "FWD"])
            position2 = c2.selectbox("Position 2", ["None", "DEF", "MID", "FWD"])
            position3 = c3.selectbox("Position 3", ["None", "DEF", "MID", "FWD"])
            st.subheader("Player Attributes (1-10)")
            c1, c2, c3 = st.columns(3)
            technical_skill = c1.slider("Technical Skill", 1.0, 10.0, 5.0)
            physical_strength = c1.slider("Physical Strength", 1.0, 10.0, 5.0)
            stamina = c2.slider("Stamina", 1.0, 10.0, 5.0)
            speed = c2.slider("Speed", 1.0, 10.0, 5.0)
            shot_accuracy = c3.slider("Shot Accuracy", 1.0, 10.0, 5.0)
            experience = c3.slider("Experience", 1.0, 10.0, 5.0)
            submitted = st.form_submit_button("Add Player")
            if submitted and name.strip():
                conn = db.get_connection()
                try:
                    if db.add_player(conn, name.strip(), position1, position2 if position2 != "None" else "", position3 if position3 != "None" else "", 
                                    technical_skill, physical_strength, stamina, speed, shot_accuracy, experience):
                        st.success(f"✅ {name} added!")
                        st.rerun()
                    else: st.error("❌ Player with this name already exists!")
                finally: conn.close()
    
    st.subheader("Current Players")
    players_df = db.get_all_players()
    if not players_df.empty:
        # --- CHANGE: Added 'games_drawn' to the displayed columns ---
        st.dataframe(players_df[['name', 'position1', 'games_won', 'games_lost', 'games_drawn', 'goals', 'assists', 'saves']], hide_index=True)

    if st.session_state.get('is_admin'):
        st.markdown("---")
        st.subheader("⚠️ Admin Zone: Deactivate Player")
        all_player_names = db.get_all_players()['name'].tolist()
        player_to_delete = st.selectbox("Select a player to deactivate", [""] + all_player_names, key="deactivate_select")
        if player_to_delete:
            st.warning(f"This will make **{player_to_delete}** unavailable for new games.")
            if st.button(f"Deactivate {player_to_delete}", type="primary"):
                db.deactivate_player(player_to_delete)
                st.success(f"{player_to_delete} has been deactivated.")
                st.rerun()

def player_stats_page(db):
    st.header("Player Statistics")
    players_df = db.get_all_players()
    if players_df.empty:
        st.info("No players found. Add players first!")
        return
    st.subheader("Overall Career Statistics")
    # --- CHANGE: Updated total_games calculation to include draws ---
    players_df['total_games'] = players_df['games_won'] + players_df['games_lost'] + players_df['games_drawn']
    players_df['win_rate'] = (players_df['games_won'] / players_df['total_games'].replace(0, 1) * 100)
    # --- CHANGE: Added 'games_drawn' to the displayed columns ---
    st.dataframe(players_df[['name', 'total_games', 'games_won', 'games_lost', 'games_drawn', 'win_rate', 'goals', 'assists', 'saves']],
        use_container_width=True, hide_index=True,
        column_config={ "win_rate": st.column_config.ProgressColumn("Win Rate",format="%.1f%%",min_value=0,max_value=100)})

    st.markdown("---")
    st.subheader("Individual Player Analysis")
    player_names = players_df['name'].tolist()
    selected_player = st.selectbox("Select a player to see their recent form:", [""] + player_names)
    if selected_player:
        history_df = db.get_player_game_history(selected_player)
        if history_df.empty:
            st.warning(f"{selected_player} hasn't played any games yet.")
        else:
            # --- CHANGE: Updated get_result function to handle draws ---
            def get_result(row):
                if row['team_a_score'] == row['team_b_score']:
                    return "Draw"
                player_is_in_team_a = selected_player in row['team_a_players']
                team_a_won = row['team_a_score'] > row['team_b_score']
                return "Win" if (player_is_in_team_a and team_a_won) or (not player_is_in_team_a and not team_a_won) else "Loss"
            history_df['result'] = history_df.apply(get_result, axis=1)
            history_df['game_label'] = pd.to_datetime(history_df['date']).dt.strftime('%d %b') + " (" + history_df['result'] + ")"
            st.markdown(f"**Recent Form for {selected_player} (Last {len(history_df)} Games)**")
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Goals', x=history_df['game_label'], y=history_df['goals']))
            fig.add_trace(go.Bar(name='Assists', x=history_df['game_label'], y=history_df['assists']))
            fig.add_trace(go.Bar(name='Saves', x=history_df['game_label'], y=history_df['saves']))
            fig.update_layout(barmode='group', title=f"Performance in Last {len(history_df)} Games")
            st.plotly_chart(fig, use_container_width=True)

def games_history_page(db):
    st.header("Games History")
    games_df = db.get_games_history()
    if games_df.empty:
        st.info("No games played yet. 🎮")
        return
    st.subheader("All Games")
    for _, game in games_df.iterrows():
        with st.expander(f"Game #{game['id']} - {game['date']} (Team A: {game['team_a_score']} - Team B: {game['team_b_score']})"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔴 Team A Players:**")
                for player in game['team_a_players'].split(','): st.write(f"• {player}")
            with col2:
                st.markdown("**🔵 Team B Players:**")
                for player in game['team_b_players'].split(','): st.write(f"• {player}")
            if st.session_state.get('is_admin'):
                st.markdown("---")
                st.warning("⚠️ Deleting a game is permanent.")
                if st.button("🗑️ Delete this Game", key=f"delete_{game['id']}"):
                    db.delete_game(game['id'])
                    st.success(f"Game #{game['id']} deleted and stats reverted.")
                    st.rerun()

def main():
    db = FootballDatabase()
    db.init_database()
    balancer = TeamBalancer()
    
    st.sidebar.title("Admin")
    admin_pass_secret = st.secrets.get("ADMIN_PASSWORD", "")
    if admin_pass_secret:
        admin_password = st.sidebar.text_input("Admin Login", type="password", key="admin_pass")
        if admin_password == admin_pass_secret:
            st.session_state['is_admin'] = True
    
    if st.session_state.get('is_admin'):
        st.sidebar.success("Admin mode unlocked!")

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["🏠 Create Game", "👥 Manage Players", "📊 Player Stats", "🏆 Games History"])
    
    if page == "🏠 Create Game":
        create_game_page(db, balancer)
    elif page == "👥 Manage Players":
        manage_players_page(db)
    elif page == "📊 Player Stats":
        player_stats_page(db)
    elif page == "🏆 Games History":
        games_history_page(db)

if __name__ == "__main__":
    main()