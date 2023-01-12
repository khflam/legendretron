import types, sys, os, decorator, inspect, argparse, copy, gzip, gc, pickle, datetime, time, psutil, pickletools, io, socket
import numpy as np
from gridtools import paths

_absprefix = os.path.abspath(sys.prefix)
virtualenv = _absprefix.split('/')[-1]

job_id_vars = ['SLURM_JOB_ID', 'JOB_ID']
is_cluster_job = False
cluster_job_id = None
for job_id_var in job_id_vars:
    cluster_job_id = os.environ.get(job_id_var, None)
    if cluster_job_id is not None:
        is_cluster_job = True
        break
cluster_job_jome = os.environ.get('CLUSTER_JOB_HOME', None)
this_hostname = socket.gethostname()
print('this_hostname', this_hostname)
print('is_cluster_job', is_cluster_job)
print('cluster_job_id', cluster_job_id)

_known_host_substrings = ('bracewell', )

mac_str, linux_str, win_str = 'osx', 'linux', 'win32'

if sys.platform == "linux" or sys.platform == "linux2":
    platform_str = linux_str
elif sys.platform == "darwin":
    platform_str = mac_str
elif sys.platform == "win32":
    platform_str = win_str
else:
    raise Exception(sys.platform)

def is_iterable(x):
    if isinstance(x, str):
        return False
    try:
        _ = iter(x)
        return True
    except TypeError:
        return False
    raise Exception(str(x))


def pickle_proof_dict(d):
    rval = dict()
    for k, v in d.items():
        try:
            pickle.dumps(v)
            okay = True
        except:
            print('pickle_proof_dict removed %s' % k)
            okay = False
        if okay:
            rval[k] = v
    return rval


def dump(x, filename, opener=open, optimize=False):
    # if optimize:
    #     raise Exception('optimize not tested with dill')
    gc.collect()
    filename = os.path.expanduser(filename)
    paths.safe_mkdir(os.path.dirname(filename))
    with opener(filename, 'wb') as fp:
        if optimize:
            s = pickle.dumps(x, pickle.HIGHEST_PROTOCOL)
            s = pickletools.optimize(s)
            fp.write(s)
        else:
            pickle.dump(x, fp, pickle.HIGHEST_PROTOCOL)
    return filename


def load(filename, exts = ('', '.gz')):
    t0 = time.time()
    filename = os.path.expanduser(filename)
    tbs = list()
    for ext in exts:
        for opener in [open, gzip.open]:
            try:
                with opener(filename+ext, 'rb') as fp:
                    if opener == gzip.open:
                        with io.BufferedReader(fp) as fpb:
                            rval = pickle.load(fpb)
                    else:
                        rval = pickle.load(fp)
                t_load = time.time() - t0
                if t_load > 10:
                    print('loaded in %i seconds' % int(t_load))
                return rval
            except:
                import traceback
                tbs.append(traceback.format_exc())
    raise Exception('\n-------------------------------------------------\n'.join(['']+tbs+['']))


def dumpArgsToDirectory(path, datestamp=False, git_info=False):

    if is_cluster_job:
        print('dumpArgsToDirectory: is_cluster_job')
    
    args_path = paths.output('dumpArgs', basepath=path, datestamp=datestamp, print_path=True)
    
    with open('%s/args.txt' % args_path, 'w') as f:
        f.write(dumpArgs.s)
        
    argdict_filename = '%s/argdict.pickle' % args_path
    dump(pickle_proof_dict(dumpArgs.argdict), filename=argdict_filename)

    if git_info:

        gitpath = '%s/gitinfo' % args_path
        paths.safe_mkdir(gitpath)

        if paths.code_roots is not None:
            for i, code_root in enumerate(paths.code_roots):
                os.system('git -C %s rev-parse HEAD > %s/HEAD.%i' % (code_root, gitpath, i))
                os.system('git -C %s diff %s > %s/diff.%i' % (code_root, code_root, gitpath, i))
                os.system('git -C %s status %s > %s/status.%i' % (code_root, code_root, gitpath, i))


def dumpArgs(filename=None, do_print=True, abbreviation_nchars=512):
    @decorator.decorator
    def inner(func, *func_args, **func_kwargs):
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        args = func_args[:len(arg_names)]
        defaults = func.__defaults__ or ()
        args = args + defaults[len(defaults) - (func.__code__.co_argcount - len(args)):]
        params = list(zip(arg_names, args))
        args = func_args[len(arg_names):]
        if args: params.append(('args', args))
        if func_kwargs: params.append(('kwargs', func_kwargs))
        s = func.__module__ + '.' + func.__name__ + '(\n'
        s = s + ('\n'.join('  %s = %s' % (abbreviated_repr(arg_name, abbreviation_nchars), abbreviated_repr(arg, abbreviation_nchars)) for arg_name, arg in params))
        s = s + '\n)'
        dumpArgs.s = s + '\n'
        dumpArgs.argdict = dict(params)
        if do_print:
            print(s)
        if filename is not None:
            with open(filename, 'w') as f:
                f.write(dumpArgs.s)
        return func(*func_args, **func_kwargs)
    return inner


def auto_parser(variables, locals_arg=None):
    parser = argparse.ArgumentParser()
    if locals_arg is not None:
        evalfn = lambda s: eval(s, locals_arg)
    else:
        evalfn = eval
    for variable in variables:
        parser.add_argument('--%s' % variable, nargs='+', type=lambda s: evalfn(s), action='append')
    args = parser.parse_args()
    args_dict = dict()
    for variable in variables:
        val = getattr(args, variable)
        val0 = val
        if val is not None:
            assert len(val) == 1, str((variable, val0, val))
            val = val[0]
            assert len(val) == 1, str((variable, val0, val))
            val = val[0]
            setattr(args, variable, val)
            args_dict[variable] = val
        else:
            delattr(args, variable)
    return args, args_dict


def auto_parser_inspector(defaults=[], allow_missing=True, locals_arg=None):
    def inner(func):
        def wrapper(*args, **kwargs):
            assert len(args) == 0, 'args are not supported'
            arg_names = sorted(inspect.signature(func).parameters.keys())
            assert 'option_module' not in arg_names
            assert 'option_arg' not in arg_names
            _, args_dict = auto_parser(arg_names + ['option_module', 'option_arg'], locals_arg=locals_arg)
            sources = copy.deepcopy(defaults)
            if 'option_module' in kwargs:
                sources += [kwargs.pop('option_module')]
            if 'option_module' in args_dict:
                sources += [args_dict.pop('option_module')]
            if 'option_arg' in args_dict:
                assert not hasattr(constants, 'option_arg')
                constants.option_arg = args_dict.pop('option_arg')
            sources += [args_dict, kwargs]
            args_dict = overrider(sources, arg_names, allow_missing=allow_missing)
            rval = func(**args_dict)
            if isinstance(rval, types.GeneratorType):
                for r in rval:
                    return r
            else:
                return rval

        return wrapper
    return inner


def overrider(sources, variables, allow_missing):
    getters = []
    missing = object()
    for source in sources:
        if isinstance(source, str):
            def this_getter(x, thism=importlib.import_module(source)):
                rval = getattr(thism, x) if hasattr(thism, x) else missing
                return rval
        elif isinstance(source, dict):
            unmatched = [k for k in list(source.keys()) if k not in variables]
            assert len(unmatched) == 0, 'unmatched dict args: %s' % ','.join(unmatched)
            def this_getter(x, d=source):
                rval = d[x] if x in d else missing
                return rval
        else:
            raise Exception('bad source %s' % str(source))
        getters.append(this_getter)

    def getfirst(var):
        results = [this_getter(var) for this_getter in getters]
        results = [(i, _) for i, _ in enumerate(results) if _ != missing]
        if len(results) == 0:
            if allow_missing:
                return missing
            else:
                raise Exception('option %s required' % var)
        i = results[-1][0]
        result = results[-1][1]
        print('  source %i:\n    %s = %s' % (i, var, abbreviated_repr(result)))
        return result
    print('( overrider sources:')
    print('\n'.join(['  %i: %s' % (i, abbreviated_repr(s)) for i, s in enumerate(map(str, sources))]))
    print(')')
    print('( overrider results')
    rval = {var: getfirst(var) for var in variables}
    for var in list(rval.keys()):
        if rval[var] == missing:
            del rval[var]
    print(')')
    return rval


def abbreviated_repr(x, nchars=512):
    a = repr(x)
    nhalf = int(nchars / 2.0)
    b = a[:nhalf] + (' ... (%i characters skipped) ... ' % (len(a) - 2 * nhalf)) + a[-nhalf:]
    return a if len(a) < len(b) else b



class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

def pretty_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Tee(object):
    def __init__(self, path, mode='w', with_time=True, dump_args=True, compress=True):
        if is_cluster_job:
            print('Tee: is_cluster_job')
            return
        self.compress = compress
        self.path = os.path.expanduser(path)
        self.file = open(self.path, mode)
        self.std = (sys.stdout, sys.stderr)
        sys.stdout = self
        sys.stderr = self
        self.with_time = with_time
        self.calls = 0
        self.nl = True
        if dump_args:
            if hasattr(dumpArgs, 's'):
                self.write(dumpArgs.s, stdout=False)
    def write(self, data, stdout=True):
        if is_cluster_job:
            return
        self.calls += 1
        if stdout:
            self.std[0].write(data)
            self.std[0].flush()
        else:
            self.std[1].write(data)
            self.std[1].flush()
        if self.with_time:
            t = pretty_time() + ' | '
            newdata = []
            for char in data:
                if self.nl:
                    newdata += t
                self.nl = char == '\n'
                newdata.append(char)
            self.file.write(''.join(newdata))
        else:
            self.file.write(data)
        self.file.flush()
    def to_file(self, print_path = True):
        if is_cluster_job:
            return
        if print_path:
            self.write('Tee.tofile()\n')
            self.write(self.path)
            self.write('\n')
        sys.stdout = self.std[0]
        sys.stderr = self.std[1]
        self.file.close()
        if self.compress:
            os.system('gzip --best %s' % self.path)

    def flush(self):
        if is_cluster_job:
            return
        self.std[0].flush()
        self.std[1].flush()


class TeeCache(object):
    def __init__(self, path, mode='w', with_time=True, dump_args=True, compress=True):
        if is_cluster_job:
            print('TeeCache: is_cluster_job')
            return
        self.compress = compress
        self.alldata = []
        self.path=os.path.expanduser(path)
        self.std = (sys.stdout, sys.stderr)
        sys.stdout = self
        self.mode=mode
        self.with_time = with_time
        self.nl = True
        if dump_args:
            if hasattr(dumpArgs, 's'):
                self.write(dumpArgs.s, stdout=False)
    def write(self, data, stdout=True):
        if is_cluster_job:
            return
        if stdout:
            self.std[0].write(data)
            self.std[0].flush()
        else:
            self.std[1].write(data)
            self.std[1].flush()
        if self.with_time:
            t = pretty_time() + ' | '
            newdata = []
            for char in data:
                if self.nl:
                    newdata += t
                self.nl = char == '\n'
                newdata.append(char)
            self.alldata.append(''.join(newdata))
        else:
            self.alldata.append(data)
        self.std[0].flush()
        self.std[1].flush()
    def to_file(self, print_path = True):
        if is_cluster_job:
            return
        if print_path:
            self.write('Tee.tofile()\n')
            self.write(self.path)
            self.write('\n')
        with open(self.path, self.mode) as f:
            for d in self.alldata:
                f.write(d)
        sys.stdout = self.std[0]
        sys.stderr = self.std[1]
        if self.compress:
            os.system('gzip --best %s' % self.path)
    def flush(self):
        if is_cluster_job:
            return
        self.std[0].flush()
        self.std[1].flush()


def pretty_duration(t):
    if t < 1e-6:
        return '{:5.1f} ns     '.format(t*1e9)
    elif t < 1e-3:
        return '{:5.1f} us     '.format(t*1e6)
    elif t < 1:
        return '{:5.1f} ms     '.format(t*1e3)
    elif t < 60:
        return '{:5.1f} seconds'.format(t)
    elif t < 60 * 60:
        return '{:5.1f} minutes'.format(t/60.0)
    elif t < 60 * 60 * 24:
        return '{:5.1f} hours  '.format(t/60.0/60)
    else:
        return '{:5.1f} days   '.format(t/60.0/60/24)


def memodict(f):
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret 
    return memodict().__getitem__


def on_host(host, known_host_substrings=_known_host_substrings):
    return any(s in this_hostname and s in host for s in known_host_substrings)


def rsync(path, host, upload, verbose=False):
    if host is None:
        return
    if on_host(host):
        print('rsync: on host %s' % host)
        return
    verbose_flag = 'v' if verbose else ''
    if upload:
        cmd = 'cd ~; rsync -az%s --exclude \'*.pyc\' --exclude \'.git\' --relative --keep-dirlinks %s %s:' % (verbose_flag, paths.unexpand_home(path, relative_to_home=True), host)
    else:
        cmd = 'cd ~; rsync -az%s --exclude \'*.pyc\' --exclude \'.git\' --relative --keep-dirlinks %s:%s .' % (verbose_flag, host, paths.unexpand_home(path, relative_to_home=True))
    if verbose:
        print('rsync command:\n', cmd)
    os.system(cmd)


def rsync_roots(s, submit_host):
    if s == 'sync_roots':
        roots = paths.sync_roots
    elif s == 'code_roots':
        roots = paths.code_roots
    if roots is not None:
        for r in roots:
            rsync(r, submit_host, upload=True, verbose=True)


def sub_list(x, ind, as_tuple=False):
    if isinstance(x, enumerate):
        x = list(x)
    if isinstance(ind, slice):
        return x[ind]
    elif ind is None:
        ind = list(range(len(x)))
    if as_tuple:
        return tuple(x[i] for i in ind)
    else:
        return [x[i] for i in ind]


def sub_list_assign(x, ix, v):
    if isinstance(ix, slice):
        x[ix] = v
    if not is_iterable(v):
        for i, thisix in enumerate(ix):
            x[thisix] = v
    else:
        for i, thisix in enumerate(ix):
            x[thisix] = v[i]


def sub_list_complement(x, ind):
    if isinstance(ind, slice):
        ind = list(range(len(x)))[ind]
    if ind is None:
        ind = list(range(len(x)))
    complement_ind = sorted(set(range(len(x))) - set(ind))
    return [x[i] for i in complement_ind]


def ind_sub_list(x, binaryind):
    return [x[_] for _, i in enumerate(binaryind) if i]


def pretty_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def pretty_duration(t):
    if np.isnan(t):
        return 'nan          '
    elif t < 1e-6:
        return '{:5.1f} ns     '.format(t*1e9)
    elif t < 1e-3:
        return '{:5.1f} us     '.format(t*1e6)
    elif t < 1:
        return '{:5.1f} ms     '.format(t*1e3)
    elif t < 60:
        return '{:5.1f} seconds'.format(t)
    elif t < 60 * 60:
        return '{:5.1f} minutes'.format(t/60.0)
    elif t < 60 * 60 * 24:
        return '{:5.1f} hours  '.format(t/60.0/60)
    else:
        return '{:5.1f} days   '.format(t/60.0/60/24)


def pretty_number(t, strip=True):
    if strip:
        return pretty_number(t, strip=False).strip()
    if np.isnan(t):
        return 'nan      '
    elif np.isinf(t):
        return 'inf      '
    elif t < 1e-6:
        return '{:5.2f} n '.format(t*1e9)
    elif t < 1e-3:
        return '{:5.2f} u '.format(t*1e6)
    elif t < 1:
        return '{:5.2f} m '.format(t*1e3)
    elif t < 1e3:
        return '{:5.2f}   '.format(t)
    elif t < 1e6:
        return '{:5.2f} K '.format(t * 1e-3)
    elif t < 1e9:
        return '{:5.2f} M '.format(t * 1e-6)
    elif t < 1e12:
        return '{:5.2f} B '.format(t * 1e-9)
    else:
        return '{:5.2e}   '.format(t)


class Timer(object):
    def __init__(self, label=''):
        self.t0 = time.time()
        self.label = label
    def __call__(self, label='', do_print=True):
        dt = time.time() - self.t0
        if do_print:
            print('%s%s%s' % (self.label, label, pretty_duration(dt)))


class ProgressBar(object):
    def __init__(self, n, freq, tfreq, ett_min_print=0, label=None):
        self.t0 = time.time()
        self.freq = freq
        self.tfreq = tfreq
        self.ett_min_print = ett_min_print
        self.i = 0
        self.tm1 = self.t0
        self.label = '(%s)' % label if label is not None else ''
        if is_iterable(n):
            self.cumincs = cumsum(n)
        else:
            self.cumincs = list(range(n))
        # print('xx', n, self.cumincs)
        self.n = self.cumincs[-1] if n > 0 else n
    def __call__(self, i=None, inc=None, label='', force_print=False):
        if self.n == 0:
            return
        if i is None:
            self.i += 1
        if inc is not None:
            self.i += inc
        if self.cumincs is None:
            i = self.i
        else:
            i = self.cumincs[min(self.i, len(self.cumincs)-1)]
        if i % self.freq == 0 or force_print:
            t = time.time()
            if i == 0:
                eta = np.nan
                ett = np.nan
            else:
                dt = t - self.t0
                r = i / float(self.n)
                eta = dt / r * (1-r)
                ett = dt / r
            if np.isnan(ett):
                ett = 0.0
            if force_print or ((t - self.tm1) >= self.tfreq and ett >= self.ett_min_print):
                mempct = memory_usage(do_print=False)['pct']
                self.tm1 = t
                rate = i / (t - self.t0)
                print('{:s}{:s} : {:s} Hz : ETA {:s} : ETT {:s} : {:12d} / {:12d} : {:s} : MEM {:4.1f}%'.format(self.label, label, pretty_number(rate), pretty_duration(eta), pretty_duration(ett), i, self.n, str(datetime.datetime.now().replace(microsecond=0)), mempct))
    def final(self):
        self(i=0, inc=0, label='(final)', force_print=True)


class progress_barred(object):
    def __init__(self, *args, **kwargs):
        self.pb = ProgressBar(*args, **kwargs)
    def __call__(self, f):
        self.f = f
        return self.g
    def g(self, *args, **kwargs):
        self.pb()
        return self.f(*args, **kwargs)


def memory_usage(s='', do_print=True, min_pct_print=-999):
    gc.collect()
    current_process = psutil.Process(os.getpid())
    d = {k + '_MB': v / 1024. / 1024. for k, v in list(dict(current_process.memory_info()._asdict()).items())}
    d['pct'] = current_process.memory_percent()
    if do_print and d['pct'] > min_pct_print:
        print('memory_usage(%s) = %s' % (s, str({k: pretty_number(int(v)) for k, v in list(d.items())})))
    return d


def set_gpu_device(x):
    if x is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(x)


def ftest_v2dict(x):
    return {'sum': sum(x), 'len': len(x)}
