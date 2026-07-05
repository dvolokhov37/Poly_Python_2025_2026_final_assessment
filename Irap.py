from AbstractMap import AbstractMap, SimpleMap
import numpy as np


class Irap(AbstractMap):

    def __init__(self, path: str | list[str]):
        super(Irap, self).__init__(path)
        self.id_irap = None
        self.rot_angle = None
        self.roto_x = None
        self.roto_y = None
        self.row_to_split = 4

    def load(self) -> SimpleMap | list[SimpleMap]:
        self.map = self.read_file()
        return self.map

    def read_file(self):
        if isinstance(self.path, str):
            with open(self.path, "r") as f:
                file = [line for line in f]
            all_rows_file = [
                file[_].split() if _ < self.row_to_split else
                [tmp if (tmp := float(v)) and (tmp <= (9999900.0 - 1.0)) else np.nan for v in file[_].split()]
                for _ in range(len(file))
            ]
            ret = self.__read_rows(all_rows_file)
            return self.preprocessing(ret)
        elif isinstance(self.path, (list, tuple, set, frozenset)):
            if len(self.path) > 0:
                maps = [Irap(_) for _ in self.path]
                return [_.read_file() for _ in maps]
            else:
                return []
        else:
            raise TypeError("Невозможно прочитать файлы в данном формате")

    def __read_rows(self, all_rows_file):
        setting = self.__read_setting_rows(all_rows_file[:self.row_to_split])
        val_map = []
        for r in all_rows_file[self.row_to_split:]:
            val_map.extend(r)

        ret = setting | dict(
            val_map=np.asarray(val_map).reshape((setting["n_col"], setting["n_row"])),
        )
        return ret

    @staticmethod
    def __read_setting_rows(rows_file):
        line = rows_file[0]
        id_map, n_col, x_inc, y_inc = int(line[0]), int(line[1]), float(line[2]), float(line[3])
        line = rows_file[1]
        x_min, x_max, y_min, y_max = float(line[0]), float(line[1]), float(line[2]), float(line[3])
        line = rows_file[2]
        n_row, rot_angle, roto_x, roto_y = int(line[0]), float(line[1]), float(line[2]), float(line[3])
        annotations = rows_file[3]
        x = np.linspace(x_min, x_max, n_row)
        y = np.linspace(y_min, y_max, n_col)
        ret = dict(
            id_map=id_map,
            n_col=n_col,
            n_row=n_row,
            x=x,
            y=y,
            rot_angle=rot_angle,
            roto_x=roto_x,
            roto_y=roto_y,
            annotations=annotations
        )
        return ret

    def preprocessing(self, ret):
        # TODO: Тут наверняка надо еще повороты наделать
        self.id_irap = ret["id_map"]
        self.rot_angle = ret["rot_angle"]
        self.roto_x = ret["roto_x"]
        self.roto_y = ret["roto_y"]
        other = dict(
            id_map=self.id_irap,
            rot_angle=self.rot_angle,
            roto_x=self.roto_x,
            roto_y=self.roto_y
        )
        # print(f"Irap:   z: {ret['val_map'].shape}\n"
        #       f"        x: {ret['x'].shape[0]}, {ret['x'].min()}, {ret['x'].max()}\n"
        #       f"        y: {ret['y'].shape[0]}, {ret['y'].min()}, {ret['y'].max()}")
        return SimpleMap(
            path=self.path,
            x=ret["x"],
            y=ret["y"],
            value=ret["val_map"],
            other=other
        )