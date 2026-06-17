from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class AlphaDiscovery:
    def __init__(self, model_type: str = "xgboost", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def _build_model(self):
        if self.model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                random_state=self.random_state, eval_metric="logloss",
            )
        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                random_state=self.random_state, verbose=-1,
            )
        elif self.model_type == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                iterations=100, depth=4, learning_rate=0.05,
                random_state=self.random_state, verbose=False,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["sma_10"] = df["close"].rolling(10).mean()
        df["sma_30"] = df["close"].rolling(30).mean()
        df["ema_12"] = df["close"].ewm(span=12).mean()
        df["ema_26"] = df["close"].ewm(span=26).mean()
        df["rsi"] = self._rsi(df["close"])
        df["volume_sma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        df["high_low_pct"] = (df["high"] - df["low"]) / df["close"]
        df["close_open_pct"] = (df["close"] - df["open"]) / df["open"]
        df["bb_upper"] = df["sma_10"] + 2 * df["close"].rolling(10).std()
        df["bb_lower"] = df["sma_10"] - 2 * df["close"].rolling(10).std()
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["atr"] = self._atr(df)
        df["target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)
        return df.dropna()

    def _rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift(1)).abs()
        lc = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def discover(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        features: List[str] = None,
    ) -> Dict:
        prepared = self.prepare_features(data)
        if features is None:
            features = ["returns", "log_returns", "sma_10", "sma_30", "ema_12", "ema_26",
                        "rsi", "volume_ratio", "high_low_pct", "close_open_pct", "bb_position", "atr"]
        self.feature_names = [f for f in features if f in prepared.columns]
        X = prepared[self.feature_names].values
        y = prepared["target"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, shuffle=False,
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        self.model = self._build_model()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        feature_importance = self._get_importance()
        return {
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "feature_importance": feature_importance,
            "feature_names": self.feature_names,
        }

    def _get_importance(self) -> Dict[str, float]:
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        prepared = self.prepare_features(data)
        X = prepared[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        prepared = self.prepare_features(data)
        X = prepared[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
