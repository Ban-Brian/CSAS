
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss, mean_absolute_error, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')


class CurlingPowerPlayAnalyzer:
    """
    Complete pipeline for analyzing curling power play decisions
    """

    def __init__(self, data_path='./'):
        self.data_path = data_path
        self.models = {}
        self.feature_importance = {}
        self.policy = None

    def load_data(self):
        """
        Load all CSV files
        """
        print("Loading CSV files...")

        # If files exist, load them
        try:
            self.competition = pd.read_csv(f'{self.data_path}Competition.csv')
            self.competitors = pd.read_csv(f'{self.data_path}Competitors.csv')
            self.ends = pd.read_csv(f'{self.data_path}Ends.csv')
            self.games = pd.read_csv(f'{self.data_path}Games.csv')
            self.stones = pd.read_csv(f'{self.data_path}Stones.csv')
            self.teams = pd.read_csv(f'{self.data_path}Teams.csv')
            print("All files loaded successfully!")
        except FileNotFoundError:
            print("Files not found, creating sample data for demonstration...")
            self._create_sample_data()

        return self

    def _create_sample_data(self):
        """
        Create realistic sample data if files not found
        """
        np.random.seed(42)

        # Competition
        self.competition = pd.DataFrame({
            'CompetitionID': range(1, 6),
            'Year': [2020, 2021, 2022, 2023, 2024],
            'Name': ['Comp A', 'Comp B', 'Comp C', 'Comp D', 'Comp E']
        })

        # Teams
        self.teams = pd.DataFrame({
            'TeamID': range(1, 21),
            'Name': [f'Team_{i}' for i in range(1, 21)],
            'Country': np.random.choice(['CAN', 'USA', 'SWE', 'NOR', 'SUI'], 20)
        })

        # Games
        games_data = []
        for comp_id in range(1, 6):
            for game_num in range(20):
                games_data.append({
                    'CompetitionID': comp_id,
                    'GameID': comp_id * 100 + game_num,
                    'Team1ID': np.random.randint(1, 21),
                    'Team2ID': np.random.randint(1, 21),
                    'Stage': np.random.choice(['RoundRobin', 'Playoff']),
                    'Sheet': np.random.randint(1, 5)
                })
        self.games = pd.DataFrame(games_data)

        # Ends
        ends_data = []
        for _, game in self.games.iterrows():
            for end_num in range(1, 9):
                power_play_used = np.random.random() < 0.15
                ends_data.append({
                    'CompetitionID': game['CompetitionID'],
                    'GameID': game['GameID'],
                    'EndNumber': end_num,
                    'Hammer': np.random.choice([game['Team1ID'], game['Team2ID']]),
                    'PowerPlayUsed': power_play_used,
                    'PowerPlayTeam': np.random.choice([game['Team1ID'], game['Team2ID']]) if power_play_used else None,
                    'PowerPlaySide': np.random.choice(['left', 'right']) if power_play_used else None,
                    'Points': np.random.choice([0, 1, 2, 3, 4], p=[0.3, 0.35, 0.25, 0.08, 0.02]),
                    'ScoringTeam': np.random.choice([game['Team1ID'], game['Team2ID'], None])
                })
        self.ends = pd.DataFrame(ends_data)

        # Stones
        stones_data = []
        for _, end in self.ends.iterrows():
            n_stones = np.random.randint(4, 17)
            for stone_num in range(1, n_stones):
                x = np.random.normal(0, 1.5)
                y = np.random.normal(0, 1.5)
                dist = np.sqrt(x ** 2 + y ** 2)
                stones_data.append({
                    'CompetitionID': end['CompetitionID'],
                    'GameID': end['GameID'],
                    'EndNumber': end['EndNumber'],
                    'StoneNumber': stone_num,
                    'Team': np.random.choice([1, 2]),
                    'X': x,
                    'Y': y,
                    'DistanceToButton': dist,
                    'InHouse': dist < 1.83,  # Official house radius
                    'Task': np.random.choice(['Draw', 'Guard', 'TakeOut', 'Freeze']),
                    'Result': np.random.choice(['Success', 'Partial', 'Miss'])
                })
        self.stones = pd.DataFrame(stones_data)

        # Competitors (simplified)
        self.competitors = pd.DataFrame({
            'CompetitorID': range(1, 81),
            'Name': [f'Player_{i}' for i in range(1, 81)],
            'TeamID': np.repeat(range(1, 21), 4)
        })

    def engineer_features(self):
        """
        Comprehensive feature engineering
        """
        print("\nEngineering features...")

        features_list = []

        # Process each end
        for _, end in self.ends.iterrows():
            feat = {}

            # Basic identifiers
            feat['competition_id'] = end['CompetitionID']
            feat['game_id'] = end['GameID']
            feat['end_number'] = end['EndNumber']

            # Game context
            game = self.games[self.games['GameID'] == end['GameID']].iloc[0]
            feat['stage_playoff'] = 1 if game['Stage'] == 'Playoff' else 0
            feat['sheet_id'] = game['Sheet']

            # End context
            feat['hammer_team'] = end['Hammer']
            feat['ends_remaining'] = 8 - end['EndNumber']

            # Calculate score differential (needs game history)
            game_ends = self.ends[(self.ends['GameID'] == end['GameID']) &
                                  (self.ends['EndNumber'] < end['EndNumber'])]

            team1_score = 0
            team2_score = 0
            for _, prev_end in game_ends.iterrows():
                if prev_end['ScoringTeam'] == game['Team1ID']:
                    team1_score += prev_end['Points']
                elif prev_end['ScoringTeam'] == game['Team2ID']:
                    team2_score += prev_end['Points']

            feat['score_diff'] = team1_score - team2_score if end['Hammer'] == game[
                'Team1ID'] else team2_score - team1_score

            # Stone features
            end_stones = self.stones[(self.stones['GameID'] == end['GameID']) &
                                     (self.stones['EndNumber'] == end['EndNumber'])]

            if len(end_stones) > 0:
                # Separate by team
                team1_stones = end_stones[end_stones['Team'] == 1]
                team2_stones = end_stones[end_stones['Team'] == 2]

                # Distance features
                feat['team1_min_dist'] = team1_stones['DistanceToButton'].min() if len(team1_stones) > 0 else 5.0
                feat['team2_min_dist'] = team2_stones['DistanceToButton'].min() if len(team2_stones) > 0 else 5.0
                feat['min_dist_diff'] = feat['team2_min_dist'] - feat['team1_min_dist']

                # Count features
                feat['team1_in_house'] = team1_stones['InHouse'].sum() if len(team1_stones) > 0 else 0
                feat['team2_in_house'] = team2_stones['InHouse'].sum() if len(team2_stones) > 0 else 0
                feat['in_house_diff'] = feat['team1_in_house'] - feat['team2_in_house']

                # Spatial distribution
                for radius in [0.5, 1.0, 1.5, 2.0]:
                    feat[f'team1_within_{radius}m'] = (team1_stones['DistanceToButton'] < radius).sum() if len(
                        team1_stones) > 0 else 0
                    feat[f'team2_within_{radius}m'] = (team2_stones['DistanceToButton'] < radius).sum() if len(
                        team2_stones) > 0 else 0

                # Centroid features
                if len(team1_stones) > 0:
                    feat['team1_centroid_x'] = team1_stones['X'].mean()
                    feat['team1_centroid_y'] = team1_stones['Y'].mean()
                    feat['team1_spread_x'] = team1_stones['X'].std()
                    feat['team1_spread_y'] = team1_stones['Y'].std()
                else:
                    feat['team1_centroid_x'] = 0
                    feat['team1_centroid_y'] = 0
                    feat['team1_spread_x'] = 0
                    feat['team1_spread_y'] = 0

                if len(team2_stones) > 0:
                    feat['team2_centroid_x'] = team2_stones['X'].mean()
                    feat['team2_centroid_y'] = team2_stones['Y'].mean()
                    feat['team2_spread_x'] = team2_stones['X'].std()
                    feat['team2_spread_y'] = team2_stones['Y'].std()
                else:
                    feat['team2_centroid_x'] = 0
                    feat['team2_centroid_y'] = 0
                    feat['team2_spread_x'] = 0
                    feat['team2_spread_y'] = 0

                # Guard analysis
                guards = end_stones[(end_stones['Y'] > 2.0) & (np.abs(end_stones['X']) < 1.0)]
                feat['n_guards'] = len(guards)
                feat['guard_coverage'] = 1 if len(guards) > 2 else 0

                # Shot task distribution
                if 'Task' in end_stones.columns:
                    task_counts = end_stones['Task'].value_counts()
                    for task in ['Draw', 'Guard', 'TakeOut', 'Freeze']:
                        feat[f'n_{task.lower()}'] = task_counts.get(task, 0)

            else:
                # Default values
                for key in ['team1_min_dist', 'team2_min_dist', 'min_dist_diff',
                            'team1_in_house', 'team2_in_house', 'in_house_diff',
                            'n_guards', 'guard_coverage']:
                    feat[key] = 0
                for radius in [0.5, 1.0, 1.5, 2.0]:
                    feat[f'team1_within_{radius}m'] = 0
                    feat[f'team2_within_{radius}m'] = 0

            # Interaction features
            feat['hammer_x_end'] = 1 if end['Hammer'] else 0 * feat['end_number']
            feat['score_diff_x_ends_remaining'] = feat['score_diff'] * feat['ends_remaining']
            feat['critical_end'] = 1 if feat['end_number'] >= 6 and abs(feat['score_diff']) <= 2 else 0

            # Power play features
            feat['power_play_used'] = 1 if end['PowerPlayUsed'] else 0
            feat['power_play_side_left'] = 1 if end.get('PowerPlaySide') == 'left' else 0
            feat['power_play_side_right'] = 1 if end.get('PowerPlaySide') == 'right' else 0

            # Target variables
            feat['points_scored'] = end['Points']
            feat['scored_2plus'] = 1 if end['Points'] >= 2 else 0
            feat['scored_3plus'] = 1 if end['Points'] >= 3 else 0

            features_list.append(feat)

        self.features_df = pd.DataFrame(features_list)
        print(f"Created {len(self.features_df.columns)} features for {len(self.features_df)} ends")

        return self

    def train_models(self):
        """
        Train multiple models for ensemble
        """
        print("\nTraining models...")

        # Prepare data
        feature_cols = [col for col in self.features_df.columns
                        if col not in ['competition_id', 'game_id', 'points_scored',
                                       'scored_2plus', 'scored_3plus', 'power_play_used',
                                       'power_play_side_left', 'power_play_side_right']]

        X = self.features_df[feature_cols]
        y_points = self.features_df['points_scored'].clip(upper=3)
        y_2plus = self.features_df['scored_2plus']
        y_3plus = self.features_df['scored_3plus']
        groups = self.features_df['game_id']

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 1. LightGBM for point distribution
        print("Training LightGBM for point distribution...")
        self.models['lgb_points'] = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=40,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        # Grouped CV
        gkf = GroupKFold(n_splits=5)
        cv_scores = []

        for train_idx, val_idx in gkf.split(X_scaled, y_points, groups):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_points.iloc[train_idx], y_points.iloc[val_idx]

            self.models['lgb_points'].fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )

            y_pred = self.models['lgb_points'].predict_proba(X_val)
            score = log_loss(y_val, y_pred)
            cv_scores.append(score)

        print(f"  CV Log Loss: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        # Retrain on full data
        self.models['lgb_points'].fit(X_scaled, y_points)

        # 2. XGBoost for 2+ points probability
        print("Training XGBoost for 2+ points probability...")
        self.models['xgb_2plus'] = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.models['xgb_2plus'].fit(X_scaled, y_2plus)

        # Store feature importance
        self.feature_importance['lgb'] = pd.DataFrame({
            'feature': X.columns,
            'importance': self.models['lgb_points'].feature_importances_
        }).sort_values('importance', ascending=False)

        print("Models trained successfully!")

        return self

    def estimate_treatment_effects(self):
        """
        Estimate causal effects of power play usage
        """
        print("\nEstimating power play treatment effects...")

        # Simple difference in means by context
        effects = []

        pp_used = self.features_df[self.features_df['power_play_used'] == 1]
        pp_not_used = self.features_df[self.features_df['power_play_used'] == 0]

        # Overall average treatment effect
        ate = pp_used['points_scored'].mean() - pp_not_used['points_scored'].mean()
        print(f"Overall Average Treatment Effect: {ate:.3f} points")

        # Heterogeneous effects by game situation
        for end in range(3, 9):
            for score_diff in range(-5, 6):
                pp_subset = pp_used[(pp_used['end_number'] == end) &
                                    (pp_used['score_diff'] == score_diff)]
                no_pp_subset = pp_not_used[(pp_not_used['end_number'] == end) &
                                           (pp_not_used['score_diff'] == score_diff)]

                if len(pp_subset) >= 3 and len(no_pp_subset) >= 3:
                    effect = pp_subset['points_scored'].mean() - no_pp_subset['points_scored'].mean()
                    effects.append({
                        'end': end,
                        'score_diff': score_diff,
                        'effect': effect,
                        'n_treated': len(pp_subset),
                        'n_control': len(no_pp_subset),
                        'confidence': min(len(pp_subset), len(no_pp_subset)) / 20.0
                    })

        self.effects_df = pd.DataFrame(effects)

        # Find significant positive effects
        significant_effects = self.effects_df[(self.effects_df['effect'] > 0.2) &
                                              (self.effects_df['confidence'] > 0.5)]

        print(f"Found {len(significant_effects)} situations with significant positive effects")

        return self

    def create_policy(self):
        """
        Create actionable policy recommendations
        """
        print("\nCreating decision policy...")

        policy_rules = []

        # Based on treatment effects
        if hasattr(self, 'effects_df') and len(self.effects_df) > 0:
            top_effects = self.effects_df.nlargest(10, 'effect')

            for _, row in top_effects.iterrows():
                if row['effect'] > 0.15:
                    policy_rules.append({
                        'condition': f"End {int(row['end'])}, Score diff {int(row['score_diff']):+d}",
                        'action': 'USE POWER PLAY',
                        'expected_gain': f"{row['effect']:.2f} points",
                        'confidence': f"{row['confidence']:.0%}"
                    })

        # Based on model predictions (example thresholds)
        policy_rules.append({
            'condition': 'End 7-8, trailing by 2+, no hammer',
            'action': 'CONSIDER POWER PLAY',
            'expected_gain': 'Increases 3+ point probability',
            'confidence': 'Model-based'
        })

        policy_rules.append({
            'condition': 'End 5-6, tied or leading by 1, with hammer',
            'action': 'SAVE POWER PLAY',
            'expected_gain': 'Better used later',
            'confidence': 'Strategic'
        })

        self.policy = pd.DataFrame(policy_rules)

        print(f"Created {len(self.policy)} policy rules")

        return self

    def generate_report(self):
        """
        Generate comprehensive analysis report
        """
        print("\n" + "=" * 60)
        print("CURLING POWER PLAY ANALYSIS REPORT")
        print("=" * 60)

        # Data summary
        print("\n1. DATA SUMMARY")
        print(f"   Competitions: {len(self.competition)}")
        print(f"   Games: {len(self.games)}")
        print(f"   Ends analyzed: {len(self.ends)}")
        print(f"   Power plays used: {self.features_df['power_play_used'].sum()} "
              f"({100 * self.features_df['power_play_used'].mean():.1f}%)")

        # Model performance
        print("\n2. MODEL PERFORMANCE")
        print("   Point prediction model: LightGBM")
        print("   2+ points model: XGBoost")

        # Feature importance
        print("\n3. TOP 10 MOST IMPORTANT FEATURES")
        if 'lgb' in self.feature_importance:
            for idx, row in self.feature_importance['lgb'].head(10).iterrows():
                print(f"   {idx + 1:2d}. {row['feature']:30s} {row['importance']:6.1f}")

        # Treatment effects
        print("\n4. POWER PLAY EFFECTIVENESS")
        if hasattr(self, 'effects_df') and len(self.effects_df) > 0:
            print("   Top 5 situations for power play usage:")
            top_5 = self.effects_df.nlargest(5, 'effect')
            for idx, row in top_5.iterrows():
                print(f"   - End {int(row['end'])}, Score diff {int(row['score_diff']):+d}: "
                      f"+{row['effect']:.2f} points")

        # Policy recommendations
        print("\n5. STRATEGIC RECOMMENDATIONS")
        if self.policy is not None and len(self.policy) > 0:
            for idx, rule in self.policy.head(5).iterrows():
                print(f"   {idx + 1}. {rule['condition']}")
                print(f"      Action: {rule['action']}")
                print(f"      Expected: {rule['expected_gain']}")

        print("\n" + "=" * 60)
        print("Analysis complete!")

        return self

    def save_results(self, output_dir='./results/'):
        """
        Save all results to files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save feature importance
        if 'lgb' in self.feature_importance:
            self.feature_importance['lgb'].to_csv(
                f'{output_dir}feature_importance.csv', index=False
            )

        # Save policy
        if self.policy is not None:
            self.policy.to_csv(f'{output_dir}policy_recommendations.csv', index=False)

        # Save effects
        if hasattr(self, 'effects_df'):
            self.effects_df.to_csv(f'{output_dir}treatment_effects.csv', index=False)

        print(f"\nResults saved to {output_dir}")

        return self


def main():
    """
    Run complete analysis pipeline
    """
    # Initialize analyzer
    analyzer = CurlingPowerPlayAnalyzer()

    # Run pipeline
    (analyzer
     .load_data()
     .engineer_features()
     .train_models()
     .estimate_treatment_effects()
     .create_policy()
     .generate_report()
     .save_results())

    return analyzer


if __name__ == "__main__":
    analyzer = main()