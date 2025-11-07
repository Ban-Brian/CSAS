import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample data matching the expected structure"""
    np.random.seed(42)
    n_games = 100
    n_ends = n_games * 8

    # Games data
    games = pd.DataFrame({
        'CompetitionID': np.repeat(range(1, 11), n_games // 10),
        'GameID': range(1, n_games + 1),
        'Team1ID': np.random.randint(1, 20, n_games),
        'Team2ID': np.random.randint(1, 20, n_games),
        'Winner': np.random.randint(1, 3, n_games)
    })

    # Ends data
    ends_data = []
    for game_id in range(1, n_games + 1):
        for end_num in range(1, 9):
            ends_data.append({
                'GameID': game_id,
                'EndNumber': end_num,
                'ScoringTeam': np.random.randint(1, 3),
                'Points': np.random.randint(0, 4),
                'PowerPlayUsed': np.random.randint(0, 2),
                'PowerPlayTeam': np.random.randint(1, 3) if np.random.random() > 0.7 else 0,
                'Hammer': np.random.randint(1, 3)
            })
    ends = pd.DataFrame(ends_data)

    # Stones data (simplified)
    stones_data = []
    for _, end in ends.iterrows():
        n_stones = np.random.randint(0, 16)
        for stone_idx in range(n_stones):
            stones_data.append({
                'GameID': end['GameID'],
                'EndNumber': end['EndNumber'],
                'StoneNumber': stone_idx + 1,
                'Team': np.random.randint(1, 3),
                'X': np.random.uniform(-2, 2),
                'Y': np.random.uniform(-2, 2),
                'InHouse': np.random.randint(0, 2)
            })
    stones = pd.DataFrame(stones_data)

    return games, ends, stones


def engineer_features(games, ends, stones):
    """
    Feature engineering following the strategy document
    """
    features = []

    for _, end in ends.iterrows():
        game_id = end['GameID']
        end_number = end['EndNumber']

        # Get stones for this end
        end_stones = stones[(stones['GameID'] == game_id) &
                            (stones['EndNumber'] == end_number)]

        # Basic features
        feat = {
            'game_id': game_id,
            'end_number': end_number,
            'hammer': end['Hammer'],
            'power_play_available': 1,  # Simplified
        }

        # Score differential (would need full game context)
        feat['score_diff'] = np.random.randint(-5, 6)  # Placeholder

        # Spatial features from stones
        if len(end_stones) > 0:
            # Distance to button (center)
            end_stones['dist_to_button'] = np.sqrt(end_stones['X'] ** 2 + end_stones['Y'] ** 2)

            team1_stones = end_stones[end_stones['Team'] == 1]
            team2_stones = end_stones[end_stones['Team'] == 2]

            # Minimum distance to button for each team
            feat['team1_min_dist'] = team1_stones['dist_to_button'].min() if len(team1_stones) > 0 else 5.0
            feat['team2_min_dist'] = team2_stones['dist_to_button'].min() if len(team2_stones) > 0 else 5.0

            # Count stones in house
            feat['team1_in_house'] = team1_stones['InHouse'].sum() if len(team1_stones) > 0 else 0
            feat['team2_in_house'] = team2_stones['InHouse'].sum() if len(team2_stones) > 0 else 0

            # Count stones within different radii
            for radius in [1.0, 2.0, 3.0]:
                feat[f'team1_within_{radius}m'] = (team1_stones['dist_to_button'] < radius).sum() if len(
                    team1_stones) > 0 else 0
                feat[f'team2_within_{radius}m'] = (team2_stones['dist_to_button'] < radius).sum() if len(
                    team2_stones) > 0 else 0

            # Spatial moments
            if len(team1_stones) > 0:
                feat['team1_centroid_x'] = team1_stones['X'].mean()
                feat['team1_centroid_y'] = team1_stones['Y'].mean()
                feat['team1_spread'] = team1_stones['dist_to_button'].std()
            else:
                feat['team1_centroid_x'] = 0
                feat['team1_centroid_y'] = 0
                feat['team1_spread'] = 0

            if len(team2_stones) > 0:
                feat['team2_centroid_x'] = team2_stones['X'].mean()
                feat['team2_centroid_y'] = team2_stones['Y'].mean()
                feat['team2_spread'] = team2_stones['dist_to_button'].std()
            else:
                feat['team2_centroid_x'] = 0
                feat['team2_centroid_y'] = 0
                feat['team2_spread'] = 0
        else:
            # Default values if no stones
            for key in ['team1_min_dist', 'team2_min_dist', 'team1_in_house', 'team2_in_house',
                        'team1_within_1.0m', 'team2_within_1.0m', 'team1_within_2.0m',
                        'team2_within_2.0m', 'team1_within_3.0m', 'team2_within_3.0m',
                        'team1_centroid_x', 'team1_centroid_y', 'team1_spread',
                        'team2_centroid_x', 'team2_centroid_y', 'team2_spread']:
                feat[key] = 0

        # Interaction features
        feat['hammer_x_end'] = feat['hammer'] * feat['end_number']
        feat['score_diff_x_ends_remaining'] = feat['score_diff'] * (8 - feat['end_number'])
        feat['in_house_diff'] = feat['team1_in_house'] - feat['team2_in_house']
        feat['min_dist_diff'] = feat['team2_min_dist'] - feat['team1_min_dist']

        # Target - points scored in this end
        feat['points'] = end['Points']
        feat['power_play_used'] = end['PowerPlayUsed']

        features.append(feat)

    return pd.DataFrame(features)


def train_baseline_model(X, y, groups):
    """
    Train LightGBM baseline model with grouped CV
    """
    # Setup grouped cross-validation
    gkf = GroupKFold(n_splits=5)

    # Define model
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=31,
        random_state=42,
        objective='multiclass',
        num_class=4,  # 0, 1, 2, 3+ points
        verbose=-1
    )

    # Cross-validation scores
    scores = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])

        y_pred_proba = model.predict_proba(X_val)
        score = log_loss(y_val, y_pred_proba)
        scores.append(score)

    print(f"Cross-validation log loss: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    # Train on full data
    model.fit(X, y)

    return model


def calculate_expected_points(model, X):
    """
    Calculate expected points from probability distribution
    """
    proba = model.predict_proba(X)
    # Expected value: sum of probability * points for each outcome
    expected = (proba[:, 0] * 0 +
                proba[:, 1] * 1 +
                proba[:, 2] * 2 +
                proba[:, 3] * 3)  # Assuming 3+ averages to 3
    return expected


def estimate_power_play_effect(df, model):
    """
    Estimate the treatment effect of using power play
    Using a simple difference in means approach for demonstration
    """
    # Split by power play usage
    pp_used = df[df['power_play_used'] == 1]
    pp_not_used = df[df['power_play_used'] == 0]

    if len(pp_used) > 0 and len(pp_not_used) > 0:
        # Calculate average treatment effect
        ate = pp_used['points'].mean() - pp_not_used['points'].mean()

        # Calculate by game situation
        effects = []
        for end in [3, 4, 5, 6, 7, 8]:
            for score_diff in range(-3, 4):
                pp_subset = pp_used[(pp_used['end_number'] == end) &
                                    (pp_used['score_diff'] == score_diff)]
                no_pp_subset = pp_not_used[(pp_not_used['end_number'] == end) &
                                           (pp_not_used['score_diff'] == score_diff)]

                if len(pp_subset) > 5 and len(no_pp_subset) > 5:
                    effect = pp_subset['points'].mean() - no_pp_subset['points'].mean()
                    effects.append({
                        'end': end,
                        'score_diff': score_diff,
                        'effect': effect,
                        'n_pp': len(pp_subset),
                        'n_no_pp': len(no_pp_subset)
                    })

        effects_df = pd.DataFrame(effects)
        return ate, effects_df
    else:
        return 0, pd.DataFrame()


def create_decision_policy(model, effects_df):
    """
    Create a simple decision policy for when to use power play
    """
    policy = []

    if len(effects_df) > 0:
        # Find situations with positive effect
        positive_effects = effects_df[effects_df['effect'] > 0.2]

        for _, row in positive_effects.iterrows():
            policy.append({
                'end': row['end'],
                'score_diff_range': (row['score_diff'] - 0.5, row['score_diff'] + 0.5),
                'action': 'USE_POWER_PLAY',
                'expected_gain': row['effect']
            })

    return policy


def main():
    """
    Main pipeline for power play decision model
    """
    print("Curling Power Play Decision Model")
    print("=" * 50)

    # Load/create data
    print("\n1. Loading data...")
    games, ends, stones = create_sample_data()
    print(f"   Games: {len(games)}")
    print(f"   Ends: {len(ends)}")
    print(f"   Stones: {len(stones)}")

    # Feature engineering
    print("\n2. Engineering features...")
    df = engineer_features(games, ends, stones)
    print(f"   Features created: {df.shape[1] - 3}")  # Exclude targets and ID

    # Prepare for modeling
    feature_cols = [col for col in df.columns
                    if col not in ['game_id', 'points', 'power_play_used']]
    X = df[feature_cols]
    y = df['points'].clip(upper=3)  # Cap at 3+ points
    groups = df['game_id']

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train baseline model
    print("\n3. Training baseline LightGBM model...")
    model = train_baseline_model(X_scaled, y, groups)

    # Calculate expected points
    print("\n4. Calculating expected points...")
    df['expected_points'] = calculate_expected_points(model, X_scaled)
    print(f"   Mean expected points: {df['expected_points'].mean():.3f}")

    # Estimate power play effect
    print("\n5. Estimating power play treatment effect...")
    ate, effects_df = estimate_power_play_effect(df, model)
    print(f"   Average Treatment Effect: {ate:.3f} points")

    # Create decision policy
    print("\n6. Creating decision policy...")
    policy = create_decision_policy(model, effects_df)

    if len(policy) > 0:
        print("\n   Recommended Power Play Usage:")
        for rule in policy[:5]:  # Show top 5 rules
            print(f"   - End {rule['end']}, Score diff {rule['score_diff_range']}: "
                  f"Expected gain = {rule['expected_gain']:.3f}")

    # Feature importance
    print("\n7. Top 10 Most Important Features:")
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")

    print("\n" + "=" * 50)
    print("Model training complete!")

    return model, df, policy


if __name__ == "__main__":
    model, results_df, policy = main()