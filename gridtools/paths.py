import os, datetime, random, errno, glob, sys
from os.path import expanduser
import gridtools.paths

_output_path = expanduser('~/output')
_base_path = expanduser(gridtools.__path__[0])
_script_path = '%s/../scripts' % _base_path

code_roots = None
sync_roots = None


def interpreter_specific_path(s='interpreter_specific_path'):
    tmp = sys.executable.split('/')
    inds = [i for i, s in enumerate(tmp) if 'virtualenvs' in s]
    assert len(inds) == 1, (tmp, inds)
    p = '/'.join(tmp[:(inds[0]+2)]+[s,''])
    return p


def safe_mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def unexpand_home(p, relative_to_home=False, home=os.path.expanduser('~')):
    if p[:len(home)] == home:
        p = '~' + p[len(home):]
    l = None
    while len(p) != l:
        l = len(p)
        p = p.replace('//','/')
    if relative_to_home:
        p = p[2:]
    return p


def now_datestamp():
    return datetime.datetime.now().isoformat().replace('-','_').replace('T','_').replace(':','_').replace('.','_')


def output(name, basepath=None, datestamp=True, create=True, interpreter_specific=False, print_path=False, datestamp_is_subdirectory=True):
    if interpreter_specific:
        assert basepath is None
        basepath = interpreter_specific_path()
    if basepath is None:
        basepath = _output_path
    if len(basepath) > 0 and basepath[-1] != '/':
        basepath = basepath + '/'
    path = os.path.expanduser('%s%s' % (basepath, name))
    if datestamp:
        nowstr = now_datestamp()
        r = '%.9i' % random.randint(0,1e9)
        if datestamp_is_subdirectory:
            path = '%s/%s_%s' % (path, nowstr, r)
        else:
            path = '%s_%s_%s' % (path, nowstr, r)
    if create:
        safe_mkdir(path)
    if print_path:
        print('paths.output:', unexpand_home(path))
    return path


def newest_file(path_wildcard_str):
    files = glob.glob(path_wildcard_str)
    return max(files, key=os.path.getctime)
