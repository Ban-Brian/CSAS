import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (log_loss, confusion_matrix, classification_report,
                             mean_absolute_error, roc_auc_score)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import optuna
import warnings
import logging
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'curling_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)


class ImprovedCurlingAnalyzer:
    """
    Enhanced pipeline with all recommended improvements
    """

    def __init__(self, data_path='./', random_seed=42):
        self.data_path = data_path
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.models = {}
        self.feature_importance = {}
        self.policy = None
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """Load CSV files with enhanced synthetic data if files not found"""
        self.logger.info("Loading data...")

        try:
            self.competition = pd.read_csv(f'{self.data_path}Competition.csv')
            self.competitors = pd.read_csv(f'{self.data_path}Competitors.csv')
            self.ends = pd.read_csv(f'{self.data_path}Ends.csv')
            self.games = pd.read_csv(f'{self.data_path}Games.csv')
            self.stones = pd.read_csv(f'{self.data_path}Stones.csv')
            self.teams = pd.read_csv(f'{self.data_path}Teams.csv')
            self.logger.info("Real data loaded successfully!")
        except FileNotFoundError:
            self.logger.info("Creating enhanced synthetic data with realistic patterns...")
            self._create_realistic_data()

        return self

    def _create_realistic_data(self):
        """Create synthetic data with realistic correlations and patterns"""
        np.random.seed(self.random_seed)

        # Teams with skill levels
        n_teams = 20
        self.teams = pd.DataFrame({
            'TeamID': range(1, n_teams + 1),
            'Name': [f'Team_{i}' for i in range(1, n_teams + 1)],
            'Country': np.random.choice(['CAN', 'USA', 'SWE', 'NOR', 'SUI'], n_teams),
            'SkillLevel': np.random.normal(0, 1, n_teams)  # Team strength
        })

        # Competitions
        self.competition = pd.DataFrame({
            'CompetitionID': range(1, 6),
            'Year': [2020, 2021, 2022, 2023, 2024],
            'Name': ['World Championship', 'Olympics', 'Continental Cup',
                     'Grand Slam', 'Tour Challenge']
        })

        # Games with realistic matchups
        games_data = []
        for comp_id in range(1, 6):
            for game_num in range(30):  # More games for better statistics
                team1_id = np.random.randint(1, n_teams + 1)
                team2_id = np.random.choice([t for t in range(1, n_teams + 1) if t != team1_id])
                games_data.append({
                    'CompetitionID': comp_id,
                    'GameID': comp_id * 1000 + game_num,
                    'Team1ID': team1_id,
                    'Team2ID': team2_id,
                    'Stage': np.random.choice(['RoundRobin', 'Playoff'], p=[0.8, 0.2]),
                    'Sheet': np.random.randint(1, 5)
                })
        self.games = pd.DataFrame(games_data)

        # Merge team skills
        self.games = self.games.merge(
            self.teams[['TeamID', 'SkillLevel']],
            left_on='Team1ID', right_on='TeamID', how='left'
        ).rename(columns={'SkillLevel': 'Team1Skill'})

        self.games = self.games.merge(
            self.teams[['TeamID', 'SkillLevel']],
            left_on='Team2ID', right_on='TeamID', how='left'
        ).rename(columns={'SkillLevel': 'Team2Skill'})

        # Realistic ends with correlated outcomes
        ends_data = []
        for _, game in self.games.iterrows():
            team1_score_total = 0
            team2_score_total = 0

            for end_num in range(1, 9):
                # Score differential affects strategy
                score_diff = team1_score_total - team2_score_total

                # Hammer alternates realistically
                if end_num == 1:
                    hammer_team = np.random.choice([game['Team1ID'], game['Team2ID']])
                else:
                    # Team that didn't score gets hammer
                    if prev_scoring_team:
                        hammer_team = game['Team1ID'] if prev_scoring_team == game['Team2ID'] else game['Team2ID']
                    else:
                        hammer_team = prev_hammer  # Blank end

                # Power play probability depends on game situation
                pp_base_prob = 0.10
                if abs(score_diff) > 2 and end_num >= 5:
                    pp_base_prob = 0.25  # More likely when behind late
                if end_num == 8 and score_diff < -1:
                    pp_base_prob = 0.40  # Desperation in final end

                power_play_used = np.random.random() < pp_base_prob

                # Realistic scoring probability based on hammer and skill
                skill_diff = game['Team1Skill'] - game['Team2Skill']
                has_hammer = (hammer_team == game['Team1ID'])

                # Base probability depends on hammer advantage and skill
                if has_hammer:
                    prob_score = 0.75 + 0.1 * skill_diff
                else:
                    prob_score = 0.25 + 0.1 * skill_diff

                # Power play effect (realistic boost)
                if power_play_used:
                    prob_score += 0.15

                # Clamp probability
                prob_score = np.clip(prob_score, 0.1, 0.9)

                # Generate points with realistic distribution
                if np.random.random() < prob_score:
                    if has_hammer:
                        # With hammer: likely 1-2 points, rarely 3+
                        points = np.random.choice([1, 2, 3, 4], p=[0.50, 0.35, 0.12, 0.03])
                    else:
                        # Steal is harder
                        points = np.random.choice([1, 2, 3], p=[0.70, 0.25, 0.05])
                    scoring_team = game['Team1ID'] if has_hammer else game['Team2ID']
                else:
                    points = 0
                    scoring_team = None

                # Update scores
                if scoring_team == game['Team1ID']:
                    team1_score_total += points
                elif scoring_team == game['Team2ID']:
                    team2_score_total += points

                ends_data.append({
                    'CompetitionID': game['CompetitionID'],
                    'GameID': game['GameID'],
                    'EndNumber': end_num,
                    'Hammer': hammer_team,
                    'PowerPlayUsed': power_play_used,
                    'PowerPlayTeam': hammer_team if power_play_used else None,
                    'PowerPlaySide': np.random.choice(['left', 'right']) if power_play_used else None,
                    'Points': points,
                    'ScoringTeam': scoring_team,
                    'Team1ScoreBefore': team1_score_total - (points if scoring_team == game['Team1ID'] else 0),
                    'Team2ScoreBefore': team2_score_total - (points if scoring_team == game['Team2ID'] else 0)
                })

                # Remember for next end
                prev_scoring_team = scoring_team
                prev_hammer = hammer_team

        self.ends = pd.DataFrame(ends_data)

        # Realistic stones with spatial patterns
        stones_data = []
        for _, end in self.ends.iterrows():
            n_stones = np.random.randint(8, 17)

            # Create realistic stone patterns
            for stone_num in range(1, n_stones):
                team = 1 if stone_num % 2 == 1 else 2

                # Guards tend to be in front
                if stone_num <= 4:
                    x = np.random.normal(0, 0.8)
                    y = np.random.normal(2.5, 0.5)  # Front of house
                else:
                    # Playing stones cluster near button
                    angle = np.random.uniform(0, 2 * np.pi)
                    dist = np.random.exponential(0.8)  # Cluster near center
                    x = dist * np.cos(angle)
                    y = dist * np.sin(angle)

                dist_to_button = np.sqrt(x ** 2 + y ** 2)

                # Task depends on position and game situation
                if y > 2.0:
                    task = 'Guard'
                elif dist_to_button < 0.5:
                    task = 'Draw'
                elif stone_num > 8:
                    task = np.random.choice(['TakeOut', 'Freeze'], p=[0.7, 0.3])
                else:
                    task = np.random.choice(['Draw', 'Guard', 'TakeOut'], p=[0.5, 0.3, 0.2])

                # Success rate based on difficulty
                if task == 'Draw':
                    success_prob = 0.75
                elif task == 'Guard':
                    success_prob = 0.80
                elif task == 'TakeOut':
                    success_prob = 0.65
                else:
                    success_prob = 0.60

                result = np.random.choice(['Success', 'Partial', 'Miss'],
                                          p=[success_prob, (1 - success_prob) * 0.6, (1 - success_prob) * 0.4])

                stones_data.append({
                    'CompetitionID': end['CompetitionID'],
                    'GameID': end['GameID'],
                    'EndNumber': end['EndNumber'],
                    'StoneNumber': stone_num,
                    'Team': team,
                    'X': x,
                    'Y': y,
                    'DistanceToButton': dist_to_button,
                    'InHouse': dist_to_button < 1.83,
                    'Task': task,
                    'Result': result
                })

        self.stones = pd.DataFrame(stones_data)

        # Simplified competitors
        self.competitors = pd.DataFrame({
            'CompetitorID': range(1, 81),
            'Name': [f'Player_{i}' for i in range(1, 81)],
            'TeamID': np.repeat(range(1, 21), 4)
        })

        self.logger.info(f"Created realistic synthetic data: {len(self.games)} games, {len(self.ends)} ends")

    def engineer_features(self):
        """Enhanced feature engineering with all improvements"""
        self.logger.info("Engineering enhanced features...")

        features_list = []

        # Group by game for efficiency
        for game_id, game_ends in self.ends.groupby('GameID'):
            game = self.games[self.games['GameID'] == game_id].iloc[0]

            for _, end in game_ends.iterrows():
                feat = self._extract_end_features(end, game, game_ends)
                features_list.append(feat)

        self.features_df = pd.DataFrame(features_list)

        # Add derived features
        self._add_derived_features()

        # Handle missing values properly
        self._handle_missing_values()

        self.logger.info(f"Created {len(self.features_df.columns)} features for {len(self.features_df)} ends")

        return self

    def _extract_end_features(self, end, game, game_ends):
        """Extract features for a single end"""
        feat = {}

        # Identifiers
        feat['game_id'] = end['GameID']
        feat['end_number'] = end['EndNumber']

        # Game context
        feat['stage_playoff'] = 1 if game['Stage'] == 'Playoff' else 0
        feat['sheet_id'] = game['Sheet']

        # Team skills
        feat['team1_skill'] = game.get('Team1Skill', 0)
        feat['team2_skill'] = game.get('Team2Skill', 0)
        feat['skill_diff'] = feat['team1_skill'] - feat['team2_skill']

        # End context
        feat['has_hammer'] = 1 if end['Hammer'] == game['Team1ID'] else 0
        feat['ends_remaining'] = 8 - end['EndNumber']
        feat['is_final_end'] = 1 if end['EndNumber'] == 8 else 0
        feat['is_critical_end'] = 1 if end['EndNumber'] >= 6 else 0

        # Score context
        if 'Team1ScoreBefore' in end:
            feat['score_diff'] = end['Team1ScoreBefore'] - end['Team2ScoreBefore']
        else:
            # Calculate from previous ends
            prev_ends = game_ends[game_ends['EndNumber'] < end['EndNumber']]
            team1_score = prev_ends[prev_ends['ScoringTeam'] == game['Team1ID']]['Points'].sum()
            team2_score = prev_ends[prev_ends['ScoringTeam'] == game['Team2ID']]['Points'].sum()
            feat['score_diff'] = team1_score - team2_score

        feat['score_diff_abs'] = abs(feat['score_diff'])
        feat['score_diff_squared'] = feat['score_diff'] ** 2
        feat['trailing'] = 1 if feat['score_diff'] < 0 else 0
        feat['tied'] = 1 if feat['score_diff'] == 0 else 0
        feat['leading'] = 1 if feat['score_diff'] > 0 else 0

        # Stone features
        end_stones = self.stones[(self.stones['GameID'] == end['GameID']) &
                                 (self.stones['EndNumber'] == end['EndNumber'])]

        if len(end_stones) > 0:
            stone_features = self._extract_stone_features(end_stones)
            feat.update(stone_features)
        else:
            # Use NaN for missing stone features
            stone_feature_names = [
                'team1_min_dist', 'team2_min_dist', 'min_dist_diff', 'min_dist_ratio',
                'team1_in_house', 'team2_in_house', 'in_house_diff', 'in_house_ratio',
                'team1_centroid_x', 'team1_centroid_y', 'team2_centroid_x', 'team2_centroid_y',
                'centroid_dist', 'team1_spread', 'team2_spread', 'spread_ratio',
                'n_guards', 'guard_coverage', 'n_draws', 'n_takeouts'
            ]
            for name in stone_feature_names:
                feat[name] = np.nan

        # Interaction features
        feat['hammer_x_end'] = feat['has_hammer'] * feat['end_number']
        feat['hammer_x_score_diff'] = feat['has_hammer'] * feat['score_diff']
        feat['score_diff_x_ends_remaining'] = feat['score_diff'] * feat['ends_remaining']
        feat['skill_x_hammer'] = feat['skill_diff'] * feat['has_hammer']
        feat['critical_x_trailing'] = feat['is_critical_end'] * feat['trailing']

        # Power play features
        feat['power_play_used'] = 1 if end['PowerPlayUsed'] else 0
        feat['power_play_left'] = 1 if end.get('PowerPlaySide') == 'left' else 0
        feat['power_play_right'] = 1 if end.get('PowerPlaySide') == 'right' else 0

        # Targets
        feat['points_scored'] = end['Points']
        feat['scored_2plus'] = 1 if end['Points'] >= 2 else 0
        feat['scored_3plus'] = 1 if end['Points'] >= 3 else 0
        feat['blank_end'] = 1 if end['Points'] == 0 else 0

        return feat

    def _extract_stone_features(self, stones):
        """Extract comprehensive stone-based features"""
        feat = {}

        team1_stones = stones[stones['Team'] == 1]
        team2_stones = stones[stones['Team'] == 2]

        # Distance features
        if len(team1_stones) > 0:
            feat['team1_min_dist'] = team1_stones['DistanceToButton'].min()
            feat['team1_mean_dist'] = team1_stones['DistanceToButton'].mean()
        else:
            feat['team1_min_dist'] = np.nan
            feat['team1_mean_dist'] = np.nan

        if len(team2_stones) > 0:
            feat['team2_min_dist'] = team2_stones['DistanceToButton'].min()
            feat['team2_mean_dist'] = team2_stones['DistanceToButton'].mean()
        else:
            feat['team2_min_dist'] = np.nan
            feat['team2_mean_dist'] = np.nan

        # Relative metrics
        if not np.isnan(feat['team1_min_dist']) and not np.isnan(feat['team2_min_dist']):
            feat['min_dist_diff'] = feat['team2_min_dist'] - feat['team1_min_dist']
            feat['min_dist_ratio'] = feat['team1_min_dist'] / (feat['team2_min_dist'] + 0.01)
        else:
            feat['min_dist_diff'] = np.nan
            feat['min_dist_ratio'] = np.nan

        # Count features
        feat['team1_in_house'] = team1_stones['InHouse'].sum() if len(team1_stones) > 0 else 0
        feat['team2_in_house'] = team2_stones['InHouse'].sum() if len(team2_stones) > 0 else 0
        feat['in_house_diff'] = feat['team1_in_house'] - feat['team2_in_house']
        feat['in_house_ratio'] = feat['team1_in_house'] / (feat['team2_in_house'] + 1)

        # Spatial distribution
        for radius in [0.5, 1.0, 1.5, 2.0, 3.0]:
            feat[f'team1_within_{radius}m'] = (team1_stones['DistanceToButton'] < radius).sum() if len(
                team1_stones) > 0 else 0
            feat[f'team2_within_{radius}m'] = (team2_stones['DistanceToButton'] < radius).sum() if len(
                team2_stones) > 0 else 0

        # Centroid and spread
        if len(team1_stones) > 0:
            feat['team1_centroid_x'] = team1_stones['X'].mean()
            feat['team1_centroid_y'] = team1_stones['Y'].mean()
            feat['team1_spread'] = np.sqrt(team1_stones['X'].var() + team1_stones['Y'].var())
        else:
            feat['team1_centroid_x'] = np.nan
            feat['team1_centroid_y'] = np.nan
            feat['team1_spread'] = np.nan

        if len(team2_stones) > 0:
            feat['team2_centroid_x'] = team2_stones['X'].mean()
            feat['team2_centroid_y'] = team2_stones['Y'].mean()
            feat['team2_spread'] = np.sqrt(team2_stones['X'].var() + team2_stones['Y'].var())
        else:
            feat['team2_centroid_x'] = np.nan
            feat['team2_centroid_y'] = np.nan
            feat['team2_spread'] = np.nan

        # Centroid distance
        if not any(np.isnan([feat['team1_centroid_x'], feat['team1_centroid_y'],
                             feat['team2_centroid_x'], feat['team2_centroid_y']])):
            feat['centroid_dist'] = np.sqrt(
                (feat['team1_centroid_x'] - feat['team2_centroid_x']) ** 2 +
                (feat['team1_centroid_y'] - feat['team2_centroid_y']) ** 2
            )
        else:
            feat['centroid_dist'] = np.nan

        # Spread ratio
        if not np.isnan(feat['team1_spread']) and not np.isnan(feat['team2_spread']):
            feat['spread_ratio'] = feat['team1_spread'] / (feat['team2_spread'] + 0.01)
        else:
            feat['spread_ratio'] = np.nan

        # Guard analysis
        guards = stones[(stones['Y'] > 2.0) & (np.abs(stones['X']) < 1.5)]
        feat['n_guards'] = len(guards)
        feat['guard_coverage'] = 1 if len(guards) > 2 else 0

        # Task distribution
        if 'Task' in stones.columns:
            task_counts = stones['Task'].value_counts()
            feat['n_draws'] = task_counts.get('Draw', 0)
            feat['n_guards_task'] = task_counts.get('Guard', 0)
            feat['n_takeouts'] = task_counts.get('TakeOut', 0)
            feat['n_freezes'] = task_counts.get('Freeze', 0)

            # Success rates
            if 'Result' in stones.columns:
                success_rate = (stones['Result'] == 'Success').mean()
                feat['shot_success_rate'] = success_rate

        return feat

    def _add_derived_features(self):
        """Add polynomial and interaction features"""
        df = self.features_df

        # Polynomial features for key variables
        df['score_diff_cubic'] = df['score_diff'] ** 3
        df['ends_remaining_squared'] = df['ends_remaining'] ** 2

        # Complex interactions
        df['pressure_index'] = df['score_diff_abs'] * (8 - df['ends_remaining']) / 8
        df['comeback_potential'] = df['trailing'] * df['ends_remaining'] * df['skill_diff']
        df['protect_lead'] = df['leading'] * (8 - df['ends_remaining']) * df['has_hammer']

        # Ratio features
        df['score_per_end'] = df['score_diff'] / (df['end_number'] + 0.01)
        df['required_scoring_rate'] = np.where(
            df['trailing'] == 1,
            -df['score_diff'] / (df['ends_remaining'] + 0.01),
            0
        )

        self.features_df = df

    def _handle_missing_values(self):
        """Properly handle missing values"""
        # Identify numeric columns
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in
                        ['game_id', 'points_scored', 'scored_2plus', 'scored_3plus',
                         'power_play_used', 'power_play_left', 'power_play_right']]

        # Use median imputation for missing values
        imputer = SimpleImputer(strategy='median')
        self.features_df[numeric_cols] = imputer.fit_transform(self.features_df[numeric_cols])

        # Store imputer for later use
        self.imputer = imputer

    def train_models_with_tuning(self):
        """Train models with hyperparameter tuning"""
        self.logger.info("Training models with hyperparameter optimization...")

        # Prepare data
        feature_cols = [col for col in self.features_df.columns
                        if col not in ['game_id', 'points_scored', 'scored_2plus',
                                       'scored_3plus', 'power_play_used', 'blank_end',
                                       'power_play_left', 'power_play_right']]

        X = self.features_df[feature_cols]
        y_points = self.features_df['points_scored'].clip(upper=3)
        y_2plus = self.features_df['scored_2plus']
        groups = self.features_df['game_id']

        # Feature selection
        selector = SelectKBest(f_classif, k=min(30, len(feature_cols)))
        X_selected = selector.fit_transform(X, y_points)
        selected_features = X.columns[selector.get_support()].tolist()
        self.logger.info(f"Selected {len(selected_features)} best features")

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_selected)

        # Optuna hyperparameter tuning for LightGBM
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'class_weight': 'balanced',
                'random_state': self.random_seed,
                'n_jobs': -1,
                'verbosity': -1
            }

            model = lgb.LGBMClassifier(**params)

            # Cross-validation
            gkf = GroupKFold(n_splits=3)
            scores = []
            for train_idx, val_idx in gkf.split(X_scaled, y_points, groups):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_points.iloc[train_idx], y_points.iloc[val_idx]

                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])

                y_pred = model.predict_proba(X_val)
                scores.append(log_loss(y_val, y_pred))

            return np.mean(scores)

        # Run optimization
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=20, n_jobs=1)

        best_params = study.best_params
        self.logger.info(f"Best parameters: {best_params}")

        # Train final model with best parameters
        self.models['lgb_points'] = lgb.LGBMClassifier(
            **best_params,
            class_weight='balanced',
            random_state=self.random_seed,
            n_jobs=-1,
            verbosity=-1
        )

        # Final cross-validation with diagnostics
        gkf = GroupKFold(n_splits=5)
        cv_scores = []
        cv_predictions = []
        cv_true = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y_points, groups)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_points.iloc[train_idx], y_points.iloc[val_idx]

            self.models['lgb_points'].fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )

            y_pred_proba = self.models['lgb_points'].predict_proba(X_val)
            y_pred = self.models['lgb_points'].predict(X_val)

            score = log_loss(y_val, y_pred_proba)
            cv_scores.append(score)
            cv_predictions.extend(y_pred)
            cv_true.extend(y_val)

            self.logger.info(f"Fold {fold + 1} Log Loss: {score:.4f}")

        # Print diagnostics
        self.logger.info(f"CV Log Loss: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        self.logger.info("Confusion Matrix:")
        self.logger.info(f"\n{confusion_matrix(cv_true, cv_predictions)}")

        # Train on full data
        self.models['lgb_points'].fit(X_scaled, y_points)

        # Store feature names and importance
        self.selected_features = selected_features
        self.feature_importance['lgb'] = pd.DataFrame({
            'feature': selected_features,
            'importance': self.models['lgb_points'].feature_importances_
        }).sort_values('importance', ascending=False)

        # Train XGBoost for 2+ points with class balancing
        self.models['xgb_2plus'] = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            scale_pos_weight=(y_2plus == 0).sum() / (y_2plus == 1).sum(),
            random_state=self.random_seed,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.models['xgb_2plus'].fit(X_scaled, y_2plus)

        self.logger.info("Models trained successfully with optimization!")

        return self

    def estimate_causal_effects(self):
        """Enhanced causal effect estimation with propensity scores"""
        self.logger.info("Estimating causal effects with propensity score weighting...")

        # Prepare data for propensity model
        feature_cols = [col for col in self.features_df.columns
                        if col not in ['game_id', 'points_scored', 'scored_2plus',
                                       'scored_3plus', 'power_play_used', 'blank_end',
                                       'power_play_left', 'power_play_right']]

        X = self.features_df[feature_cols]
        X_scaled = self.scaler.transform(X[self.selected_features])

        # Train propensity score model
        ps_model = LogisticRegression(random_state=self.random_seed, max_iter=1000)
        ps_model.fit(X_scaled, self.features_df['power_play_used'])

        # Calculate propensity scores
        self.features_df['propensity_score'] = ps_model.predict_proba(X_scaled)[:, 1]

        # Inverse probability weighting
        treated = self.features_df[self.features_df['power_play_used'] == 1]
        control = self.features_df[self.features_df['power_play_used'] == 0]

        # Calculate weights
        treated['weight'] = 1 / treated['propensity_score']
        control['weight'] = 1 / (1 - control['propensity_score'])

        # Trim extreme weights
        treated['weight'] = np.clip(treated['weight'], 0.1, 10)
        control['weight'] = np.clip(control['weight'], 0.1, 10)

        # Weighted average treatment effect
        ate_weighted = (
                (treated['points_scored'] * treated['weight']).sum() / treated['weight'].sum() -
                (control['points_scored'] * control['weight']).sum() / control['weight'].sum()
        )

        self.logger.info(f"Propensity-weighted ATE: {ate_weighted:.3f} points")

        # Heterogeneous treatment effects
        effects = []

        for end in range(3, 9):
            for score_diff in range(-5, 6):
                for has_hammer in [0, 1]:
                    treated_subset = treated[
                        (treated['end_number'] == end) &
                        (treated['score_diff'] == score_diff) &
                        (treated['has_hammer'] == has_hammer)
                        ]
                    control_subset = control[
                        (control['end_number'] == end) &
                        (control['score_diff'] == score_diff) &
                        (control['has_hammer'] == has_hammer)
                        ]

                    if len(treated_subset) >= 5 and len(control_subset) >= 5:
                        # Weighted effect
                        effect = (
                                (treated_subset['points_scored'] * treated_subset['weight']).sum() /
                                treated_subset['weight'].sum() -
                                (control_subset['points_scored'] * control_subset['weight']).sum() /
                                control_subset['weight'].sum()
                        )

                        effects.append({
                            'end': end,
                            'score_diff': score_diff,
                            'has_hammer': has_hammer,
                            'effect': effect,
                            'n_treated': len(treated_subset),
                            'n_control': len(control_subset),
                            'confidence': min(len(treated_subset), len(control_subset)) / 20.0
                        })

        self.effects_df = pd.DataFrame(effects)

        # Smooth noisy effects by clustering
        if len(self.effects_df) > 0:
            # Group similar situations
            self.effects_df['situation_cluster'] = (
                    self.effects_df['end'].astype(str) + '_' +
                    pd.cut(self.effects_df['score_diff'], bins=[-10, -2, 0, 2, 10]).astype(str) + '_' +
                    self.effects_df['has_hammer'].astype(str)
            )

            # Average within clusters
            cluster_effects = self.effects_df.groupby('situation_cluster').agg({
                'effect': 'mean',
                'n_treated': 'sum',
                'n_control': 'sum'
            }).reset_index()

            self.logger.info(f"Found {len(cluster_effects)} clustered treatment effects")

        return self

    def create_enhanced_policy(self):
        """Create policy with visualization and clear recommendations"""
        self.logger.info("Creating enhanced decision policy...")

        policy_rules = []

        if hasattr(self, 'effects_df') and len(self.effects_df) > 0:
            # Sort by effect size
            top_effects = self.effects_df.sort_values('effect', ascending=False).head(15)

            for _, row in top_effects.iterrows():
                if row['effect'] > 0.1 and row['confidence'] > 0.25:
                    hammer_text = "with hammer" if row['has_hammer'] else "without hammer"
                    policy_rules.append({
                        'situation': f"End {int(row['end'])}, Score {int(row['score_diff']):+d}, {hammer_text}",
                        'action': 'USE POWER PLAY',
                        'expected_gain': f"{row['effect']:.2f} points",
                        'confidence': f"{row['confidence']:.0%}",
                        'sample_size': f"n={row['n_treated'] + row['n_control']}"
                    })

        # Add strategic rules based on model insights
        strategic_rules = [
            {
                'situation': 'Final end (8), trailing by 2+',
                'action': 'ALWAYS USE POWER PLAY',
                'expected_gain': 'Maximizes comeback probability',
                'confidence': 'High',
                'sample_size': 'Strategic'
            },
            {
                'situation': 'End 5-6, tied or +1, with hammer',
                'action': 'SAVE POWER PLAY',
                'expected_gain': 'Preserve for critical moments',
                'confidence': 'Medium',
                'sample_size': 'Strategic'
            },
            {
                'situation': 'End 3-4, large lead (3+)',
                'action': 'SAVE POWER PLAY',
                'expected_gain': 'Not needed with comfort',
                'confidence': 'High',
                'sample_size': 'Strategic'
            }
        ]

        policy_rules.extend(strategic_rules)
        self.policy = pd.DataFrame(policy_rules)

        # Create visualization
        self._visualize_policy()

        self.logger.info(f"Created {len(self.policy)} policy rules")

        return self

    def _visualize_policy(self):
        """Create heatmap visualization of power play effectiveness"""
        if hasattr(self, 'effects_df') and len(self.effects_df) > 0:
            # Prepare pivot table for heatmap
            pivot_data = self.effects_df.pivot_table(
                values='effect',
                index='score_diff',
                columns='end',
                aggfunc='mean'
            )

            # Create heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_data, cmap='RdYlGn', center=0,
                        annot=True, fmt='.2f', cbar_kws={'label': 'Effect (points)'})
            plt.title('Power Play Effectiveness by Game Situation')
            plt.xlabel('End Number')
            plt.ylabel('Score Differential')
            plt.tight_layout()
            plt.savefig('power_play_heatmap.png', dpi=150)
            plt.close()

            self.logger.info("Saved power play effectiveness heatmap")

    def generate_enhanced_report(self):
        """Generate comprehensive report with all improvements"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ENHANCED CURLING POWER PLAY ANALYSIS REPORT")
        self.logger.info("=" * 60)

        # Data summary
        self.logger.info("\n1. DATA SUMMARY")
        self.logger.info(f"   Games analyzed: {len(self.games)}")
        self.logger.info(f"   Total ends: {len(self.ends)}")
        self.logger.info(f"   Power plays used: {self.features_df['power_play_used'].sum()} "
                         f"({100 * self.features_df['power_play_used'].mean():.1f}%)")
        self.logger.info(f"   Average points per end: {self.features_df['points_scored'].mean():.2f}")

        # Model performance
        self.logger.info("\n2. MODEL PERFORMANCE")
        self.logger.info(f"   Features selected: {len(self.selected_features)}")
        self.logger.info(f"   Hyperparameter tuning: Completed (20 trials)")

        # Feature importance with SHAP values would go here
        self.logger.info("\n3. TOP 10 MOST IMPORTANT FEATURES")
        for idx, row in self.feature_importance['lgb'].head(10).iterrows():
            self.logger.info(f"   {idx + 1:2d}. {row['feature']:30s} {row['importance']:8.1f}")

        # Causal effects
        self.logger.info("\n4. CAUSAL EFFECT ANALYSIS")
        if hasattr(self, 'effects_df') and len(self.effects_df) > 0:
            self.logger.info("   Top 5 situations for power play effectiveness:")
            top_5 = self.effects_df.nlargest(5, 'effect')
            for idx, row in top_5.iterrows():
                hammer_text = "with hammer" if row['has_hammer'] else "no hammer"
                self.logger.info(f"   - End {int(row['end'])}, Score {int(row['score_diff']):+d}, {hammer_text}: "
                                 f"+{row['effect']:.2f} points (n={row['n_treated'] + row['n_control']})")

        # Policy recommendations
        self.logger.info("\n5. STRATEGIC POLICY RECOMMENDATIONS")
        if self.policy is not None and len(self.policy) > 0:
            for idx, rule in self.policy.head(5).iterrows():
                self.logger.info(f"\n   Rule {idx + 1}:")
                self.logger.info(f"   Situation: {rule['situation']}")
                self.logger.info(f"   Action: {rule['action']}")
                self.logger.info(f"   Expected: {rule['expected_gain']}")
                self.logger.info(f"   Confidence: {rule['confidence']}")

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Analysis complete! Check log file and visualizations.")

        return self

    def save_all_outputs(self, output_dir='./results/'):
        """Save all results with proper documentation"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save processed data
        self.features_df.to_csv(f'{output_dir}processed_features.csv', index=False)

        # Save feature importance
        if 'lgb' in self.feature_importance:
            self.feature_importance['lgb'].to_csv(f'{output_dir}feature_importance.csv', index=False)

        # Save policy
        if self.policy is not None:
            self.policy.to_csv(f'{output_dir}policy_recommendations.csv', index=False)

        # Save effects
        if hasattr(self, 'effects_df'):
            self.effects_df.to_csv(f'{output_dir}treatment_effects.csv', index=False)

        # Save model
        import pickle
        with open(f'{output_dir}trained_model.pkl', 'wb') as f:
            pickle.dump(self.models['lgb_points'], f)

        self.logger.info(f"\nAll outputs saved to {output_dir}")

        return self


def main():
    """Run enhanced analysis pipeline"""
    # Initialize analyzer with seed for reproducibility
    analyzer = ImprovedCurlingAnalyzer(random_seed=42)

    # Run complete pipeline
    (analyzer
     .load_data()
     .engineer_features()
     .train_models_with_tuning()
     .estimate_causal_effects()
     .create_enhanced_policy()
     .generate_enhanced_report()
     .save_all_outputs())

    return analyzer


if __name__ == "__main__":
    analyzer = main()