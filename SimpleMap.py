from dataclasses import dataclass
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
from numpy.typing import NDArray
#import swifter


@dataclass
class SimpleMap:
    path: str
    x: NDArray
    y: NDArray
    value: NDArray
    other: dict
    knn: KNeighborsRegressor or None = None

    def add_mean(self, df_search: pd.DataFrame, r: float = 350):
        df_coords = self.to_dataframe()
        self.add_knn()
        df_search = df_search.apply(self.__mean_value, axis=1, args=(r, df_coords))
        return df_search

    def __mean_value(self, row, r: float, df_coords: pd.DataFrame):
        mask = (df_coords['x'] - row['x']) ** 2 + (df_coords['y'] - row['y']) ** 2 <= r ** 2
        df = df_coords[mask]
        if df.shape[0] > 1:
            v = df['value'].product() ** (1 / df.shape[0])
        else:
            N = 100
            x = np.linspace(row['x'] - r, r + row['x'], N)
            tmp = pd.DataFrame({
                'x': np.asarray([*x, *x]),
                'y': np.asarray([
                    *(row['y'] + np.sqrt(r ** 2 - (row['x'] - x) ** 2)),
                    *(row['y'] - np.sqrt(r ** 2 - (row['x'] - x) ** 2))
                ])
            })
            v = self.knn.predict(tmp).product()**(1/(2*N))
        row['mean'] = v
        return row

    @property
    def shape(self):
        x = np.unique(self.x)
        y = np.unique(self.y)
        return x.shape[0], y.shape[0]

    def max(self):
        df = self.to_dataframe()
        return df[df['value'].notna()]['value'].max()

    def min(self):
        df = self.to_dataframe()
        return df[df['value'].notna()]['value'].min()

    def __add__(self, other):
        if isinstance(other, SimpleMap):
            s, o = self.__intersection(other)
            df_s = s.to_dataframe().dropna(subset="value").sort_values(by=["x", "y"], inplace=False, ignore_index=True)
            r = df_s["value"].values + o.add_knn().predict(df_s[["x", "y"]])
        else:
            df_s = self.to_dataframe()
            r = df_s["value"] + other
        res = pd.DataFrame({
            "x": df_s["x"],
            "y": df_s["y"],
            "value": r
        }).dropna(subset=["x", "y"])
        return self.from_dataframe(
            df=res, other=self.other, knn=self.knn, path=self.path
        )

    def __mul__(self, other):
        if isinstance(other, SimpleMap):
            s, o = self.__intersection(other)
            df_s = s.to_dataframe().dropna(subset="value").sort_values(by=["x", "y"], inplace=False, ignore_index=True)
            r = df_s["value"].values * o.add_knn().predict(df_s[["x", "y"]])
        else:
            df_s = self.to_dataframe()
            r = df_s["value"] * other
        res = pd.DataFrame({
            "x": df_s["x"],
            "y": df_s["y"],
            "value": r
        }).dropna(subset=["x", "y"])
        return self.from_dataframe(
            df=res, other=self.other, knn=self.knn, path=self.path
        )

    def __sub__(self, other):
        if isinstance(other, SimpleMap):
            s, o = self.__intersection(other)
            df_s = s.to_dataframe().dropna(subset="value").sort_values(by=["x", "y"], inplace=False, ignore_index=True)
            r = df_s["value"].values - o.add_knn().predict(df_s[["x", "y"]])
            # r = df_s.swifter.apply(self.__sub, axis=1, args=(other_df_to_operations, ))
        else:
            df_s = self.to_dataframe()
            r = df_s["value"] - other
        res = pd.DataFrame({
            "x": df_s["x"],
            "y": df_s["y"],
            "value": r
        }).dropna(subset=["x", "y"])
        return self.from_dataframe(
            df=res, other=self.other, knn=None,  # self.knn,
            path=self.path
        )

    def __truediv__(self, other):
        if isinstance(other, SimpleMap):
            s, o = self.__intersection(other)
            df_s = s.to_dataframe().dropna(subset="value").sort_values(by=["x", "y"], inplace=False, ignore_index=True)
            r = df_s["value"].values / o.add_knn().predict(df_s[["x", "y"]])
        else:
            df_s = self.to_dataframe()
            r = df_s["value"] / other

        res = pd.DataFrame({
            "x": df_s["x"],
            "y": df_s["y"],
            "value": r.replace(np.inf, np.nan).replace(-np.inf, np.nan)
        }).dropna(subset=["x", "y"])
        return self.from_dataframe(
            df=res, other=self.other, knn=self.knn, path=self.path
        )

    def __pow__(self, other, modulo=None):
        if isinstance(other, SimpleMap):
            s, o = self.__intersection(other)
            df_s = s.to_dataframe().dropna(subset="value").sort_values(by=["x", "y"], inplace=False, ignore_index=True)
            r = df_s["value"].values ** o.add_knn().predict(df_s[["x", "y"]])
        else:
            df_s = self.to_dataframe()
            r = df_s["value"] ** other
        res = pd.DataFrame({
            "x": df_s["x"],
            "y": df_s["y"],
            "value": r
        }).dropna(subset=["x", "y"])
        return self.from_dataframe(
            df=res, other=self.other, knn=self.knn, path=self.path
        )

    def add_knn(self):
        if len(self.value.shape) > 1:
            x_, y_ = np.meshgrid(self.x, self.y, indexing='xy')
            x = x_.reshape(x_.shape[0] * x_.shape[1])
            y = y_.reshape(y_.shape[0] * y_.shape[1])
            df = pd.DataFrame(
                {
                    "x": x,
                    "y": y,
                    "value": self.value.reshape(self.value.shape[0] * self.value.shape[1])
                }
            )
        else:
            df = pd.DataFrame(
                {
                    "x": self.x,
                    "y": self.y,
                    "value": self.value
                }
            )
        # print('Исходное число: ', len(df))
        df = df[df['value'].notna()]
        # print('Без пропусков: ', len(df))
        knn = KNeighborsRegressor(n_neighbors=6, weights='distance', n_jobs=-1)
        knn.fit(df[['x', 'y']], df['value'])
        self.knn = knn
        return knn
        # knn.predict(another_df[['x', 'y']])

    def value_to_line(self):
        if len(self.value.shape) > 1:
            x_, y_ = np.meshgrid(self.x, self.y, indexing='xy')
            x = x_.reshape(x_.shape[0] * x_.shape[1])
            y = y_.reshape(y_.shape[0] * y_.shape[1])
            z = self.value.reshape(self.value.shape[0] * self.value.shape[1])
        else:
            x, y, z = self.x, self.y, self.value
        return SimpleMap(
            path=self.path,
            x=x,
            y=y,
            value=z,
            other=self.other,
            knn=self.knn
        )

    def value_to_matrix(self):
        if len(self.value.shape) == 1:
            # x = np.unique(self.x)
            # y = np.unique(self.y)
            # z = self.value.reshape((y.shape[0], x.shape[0]))

            xx = self.x
            yy = self.y
            zz = self.value
            x = np.unique(xx)
            y = np.unique(yy)
            z = np.empty((x.shape[0], y.shape[0]))
            z.fill(np.NaN)
            for i in range(len(zz)):
                x_i = np.where(x == xx[i])
                y_i = np.where(y == yy[i])
                z[x_i, y_i] = zz[i]
            z = z.transpose()

        else:
            x, y, z = self.x, self.y, self.value
        return SimpleMap(
            path=self.path,
            x=x,
            y=y,
            value=z,
            other=self.other,
            knn=self.knn
        )

    def to_dataframe(self):
        v = self.value_to_line()
        return pd.DataFrame({
            "x": v.x,
            "y": v.y,
            "value": v.value
        })

    def fill_na(self):
        # df = df[df['value'].notna()]
        df = self.to_dataframe()
        check_na = df[df['value'].isna()]
        df = df[df['value'].notna()]
        val = self.add_knn().predict(check_na[['x', 'y']])
        check_na['value'] = val
        ret_df = pd.concat([df, check_na])
        return self.from_dataframe(ret_df, self.path, self.other)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, path: str, other: dict or None = None,
                       knn: KNeighborsRegressor or None = None):
        return cls(
            x=df["x"].values,
            y=df["y"].values,
            value=df["value"].values,
            path=path,
            other=other if not other is None else dict(),
            knn=knn
        )

    def __intersection(self, other) -> tuple:
        s_shape = self.shape
        o_shape = other.shape
        if (s_shape[0] == o_shape[0]) and (s_shape[1] == o_shape[1]):
            if (sorted(self.x) == sorted(other.x)) and (sorted(self.y) == sorted(other.y)):
                return self, other
        x_min = max(self.x.min(), other.x.min())
        y_min = max(self.y.min(), other.y.min())
        x_max = min(self.x.max(), other.x.max())
        y_max = min(self.y.max(), other.y.max())
        x_not_intersection = (x_min > self.x.max()) or (x_min > other.x.max()) or (x_max < self.x.min()) or \
                             (x_max < other.x.min())
        y_not_intersection = (y_min > self.y.max()) or (y_min > other.y.max()) or (y_max < self.y.min()) or \
                             (y_max < other.y.min())
        if x_not_intersection or y_not_intersection:
            raise ValueError(f"Карты {self.path} и {other.path} не пересекаются!")
        self_df = self.to_dataframe()
        other_df = other.to_dataframe()
        mask_self = (self_df["x"] > x_min) & (self_df["x"] < x_max) & (self_df["y"] > y_min) & (self_df["y"] < y_max)
        mask_other = (other_df["x"] > x_min) & (other_df["x"] < x_max) & (other_df["y"] > y_min) & \
                     (other_df["y"] < y_max)
        ret_self = self.from_dataframe(self_df[mask_self], self.path, other=self.other, knn=self.knn)
        ret_other = self.from_dataframe(other_df[mask_other], other.path, other=other.other, knn=other.knn)
        return ret_self, ret_other
