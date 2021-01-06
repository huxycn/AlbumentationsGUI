import shutil
from pathlib import Path


class DirectoryNotEmptyError(Exception):
    def __init__(self):
        super().__init__()


def copy(src, dst):
    shutil.copy(src, dst)


def get_home_dir():
    return str(Path().home())


def get_parent_dir(path):
    return str(Path(path).parent)


def new_dir(path, rm_all=False):
    if not Path(path).exists():
        # 文件夹不存在 => 创建
        Path(path).mkdir()
    elif list(Path(path).glob('*')):
        # 文件夹存在且不为空 => 判断 rm_all
        if rm_all:
            # rm_all=True => 清空文件夹
            shutil.rmtree(path)
            Path(path).mkdir()
        else:
            # rm_all=False => 报错
            raise DirectoryNotEmptyError()
    else:
        # 文件夹存在且为空 => pass
        pass


def join(path1, path2):
    return str(Path(path1).joinpath(path2))


def get_path_with_suffix(path, suffix):
    return str(Path(path).with_suffix(suffix))


def get_name_with_suffix(path, suffix=None):
    if suffix is None:
        return Path(path).name
    return Path(path).with_suffix(suffix).name


def get_stem(path):
    return Path(path).stem


def get_suffix(path):
    return Path(path).suffix


def exists(path):
    return Path(path).exists()


def glob(path, pattern):

    if ';;' in pattern:
        ret_list = []
        pattern_list = pattern.split(';;')
        for pa in pattern_list:
            ret_list.extend([str(p) for p in Path(path).glob(pa)])
    else:
        ret_list = [str(p) for p in Path(path).glob(pattern)]
    return ret_list


def root():
    return str(Path(__file__).parent.parent)
