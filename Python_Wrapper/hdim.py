# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.8
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_hdim', [dirname(__file__)])
        except ImportError:
            import _hdim
            return _hdim
        if fp is not None:
            try:
                _mod = imp.load_module('_hdim', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _hdim = swig_import_helper()
    del swig_import_helper
else:
    import _hdim
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0



_hdim.SolverType_ista_swigconstant(_hdim)
SolverType_ista = _hdim.SolverType_ista

_hdim.SolverType_fista_swigconstant(_hdim)
SolverType_fista = _hdim.SolverType_fista

_hdim.SolverType_cd_swigconstant(_hdim)
SolverType_cd = _hdim.SolverType_cd

_hdim.SolverType_lazy_cd_swigconstant(_hdim)
SolverType_lazy_cd = _hdim.SolverType_lazy_cd
class FOS_d(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FOS_d, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FOS_d, name)
    __repr__ = _swig_repr

    def __init__(self, x, y):
        this = _hdim.new_FOS_d(x, y)
        try:
            self.this.append(this)
        except Exception:
            self.this = this

    def Algorithm(self):
        return _hdim.FOS_d_Algorithm(self)

    def ReturnLambda(self):
        return _hdim.FOS_d_ReturnLambda(self)

    def ReturnBetas(self):
        return _hdim.FOS_d_ReturnBetas(self)

    def ReturnOptimIndex(self):
        return _hdim.FOS_d_ReturnOptimIndex(self)

    def ReturnCoefficients(self):
        return _hdim.FOS_d_ReturnCoefficients(self)

    def ReturnSupport(self):
        return _hdim.FOS_d_ReturnSupport(self)
    __swig_destroy__ = _hdim.delete_FOS_d
    __del__ = lambda self: None
FOS_d_swigregister = _hdim.FOS_d_swigregister
FOS_d_swigregister(FOS_d)

class FOS_f(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FOS_f, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FOS_f, name)
    __repr__ = _swig_repr

    def __init__(self, x, y):
        this = _hdim.new_FOS_f(x, y)
        try:
            self.this.append(this)
        except Exception:
            self.this = this

    def Algorithm(self):
        return _hdim.FOS_f_Algorithm(self)

    def ReturnLambda(self):
        return _hdim.FOS_f_ReturnLambda(self)

    def ReturnBetas(self):
        return _hdim.FOS_f_ReturnBetas(self)

    def ReturnOptimIndex(self):
        return _hdim.FOS_f_ReturnOptimIndex(self)

    def ReturnCoefficients(self):
        return _hdim.FOS_f_ReturnCoefficients(self)

    def ReturnSupport(self):
        return _hdim.FOS_f_ReturnSupport(self)
    __swig_destroy__ = _hdim.delete_FOS_f
    __del__ = lambda self: None
FOS_f_swigregister = _hdim.FOS_f_swigregister
FOS_f_swigregister(FOS_f)

class X_FOS_d(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, X_FOS_d, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, X_FOS_d, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = _hdim.new_X_FOS_d()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _hdim.delete_X_FOS_d
    __del__ = lambda self: None

    def __call__(self, *args):
        return _hdim.X_FOS_d___call__(self, *args)

    def ReturnLambda(self):
        return _hdim.X_FOS_d_ReturnLambda(self)

    def ReturnIntercept(self):
        return _hdim.X_FOS_d_ReturnIntercept(self)

    def ReturnBetas(self):
        return _hdim.X_FOS_d_ReturnBetas(self)

    def ReturnOptimIndex(self):
        return _hdim.X_FOS_d_ReturnOptimIndex(self)

    def ReturnCoefficients(self):
        return _hdim.X_FOS_d_ReturnCoefficients(self)

    def ReturnSupport(self):
        return _hdim.X_FOS_d_ReturnSupport(self)
X_FOS_d_swigregister = _hdim.X_FOS_d_swigregister
X_FOS_d_swigregister(X_FOS_d)

class X_FOS_f(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, X_FOS_f, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, X_FOS_f, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = _hdim.new_X_FOS_f()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _hdim.delete_X_FOS_f
    __del__ = lambda self: None

    def __call__(self, *args):
        return _hdim.X_FOS_f___call__(self, *args)

    def ReturnLambda(self):
        return _hdim.X_FOS_f_ReturnLambda(self)

    def ReturnIntercept(self):
        return _hdim.X_FOS_f_ReturnIntercept(self)

    def ReturnBetas(self):
        return _hdim.X_FOS_f_ReturnBetas(self)

    def ReturnOptimIndex(self):
        return _hdim.X_FOS_f_ReturnOptimIndex(self)

    def ReturnCoefficients(self):
        return _hdim.X_FOS_f_ReturnCoefficients(self)

    def ReturnSupport(self):
        return _hdim.X_FOS_f_ReturnSupport(self)
X_FOS_f_swigregister = _hdim.X_FOS_f_swigregister
X_FOS_f_swigregister(X_FOS_f)

class Solver_f(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Solver_f, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Solver_f, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _hdim.delete_Solver_f
    __del__ = lambda self: None

    def __call__(self, *args):
        return _hdim.Solver_f___call__(self, *args)
Solver_f_swigregister = _hdim.Solver_f_swigregister
Solver_f_swigregister(Solver_f)

class Solver_d(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Solver_d, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Solver_d, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _hdim.delete_Solver_d
    __del__ = lambda self: None

    def __call__(self, *args):
        return _hdim.Solver_d___call__(self, *args)
Solver_d_swigregister = _hdim.Solver_d_swigregister
Solver_d_swigregister(Solver_d)

class SGD_f(Solver_f):
    __swig_setmethods__ = {}
    for _s in [Solver_f]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, SGD_f, name, value)
    __swig_getmethods__ = {}
    for _s in [Solver_f]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, SGD_f, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _hdim.delete_SGD_f
    __del__ = lambda self: None
SGD_f_swigregister = _hdim.SGD_f_swigregister
SGD_f_swigregister(SGD_f)

class SGD_d(Solver_d):
    __swig_setmethods__ = {}
    for _s in [Solver_d]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, SGD_d, name, value)
    __swig_getmethods__ = {}
    for _s in [Solver_d]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, SGD_d, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _hdim.delete_SGD_d
    __del__ = lambda self: None
SGD_d_swigregister = _hdim.SGD_d_swigregister
SGD_d_swigregister(SGD_d)

class ISTA_f(SGD_f):
    __swig_setmethods__ = {}
    for _s in [SGD_f]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, ISTA_f, name, value)
    __swig_getmethods__ = {}
    for _s in [SGD_f]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, ISTA_f, name)
    __repr__ = _swig_repr

    def __init__(self, L_0=0.1):
        this = _hdim.new_ISTA_f(L_0)
        try:
            self.this.append(this)
        except Exception:
            self.this = this

    def __call__(self, *args):
        return _hdim.ISTA_f___call__(self, *args)
    __swig_destroy__ = _hdim.delete_ISTA_f
    __del__ = lambda self: None
ISTA_f_swigregister = _hdim.ISTA_f_swigregister
ISTA_f_swigregister(ISTA_f)

class ISTA_d(SGD_d):
    __swig_setmethods__ = {}
    for _s in [SGD_d]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, ISTA_d, name, value)
    __swig_getmethods__ = {}
    for _s in [SGD_d]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, ISTA_d, name)
    __repr__ = _swig_repr

    def __init__(self, L_0=0.1):
        this = _hdim.new_ISTA_d(L_0)
        try:
            self.this.append(this)
        except Exception:
            self.this = this

    def __call__(self, *args):
        return _hdim.ISTA_d___call__(self, *args)
    __swig_destroy__ = _hdim.delete_ISTA_d
    __del__ = lambda self: None
ISTA_d_swigregister = _hdim.ISTA_d_swigregister
ISTA_d_swigregister(ISTA_d)

class FISTA_f(SGD_f):
    __swig_setmethods__ = {}
    for _s in [SGD_f]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, FISTA_f, name, value)
    __swig_getmethods__ = {}
    for _s in [SGD_f]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, FISTA_f, name)
    __repr__ = _swig_repr

    def __init__(self, L_0=0.1):
        this = _hdim.new_FISTA_f(L_0)
        try:
            self.this.append(this)
        except Exception:
            self.this = this

    def __call__(self, *args):
        return _hdim.FISTA_f___call__(self, *args)
    __swig_destroy__ = _hdim.delete_FISTA_f
    __del__ = lambda self: None
FISTA_f_swigregister = _hdim.FISTA_f_swigregister
FISTA_f_swigregister(FISTA_f)

class FISTA_d(SGD_d):
    __swig_setmethods__ = {}
    for _s in [SGD_d]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, FISTA_d, name, value)
    __swig_getmethods__ = {}
    for _s in [SGD_d]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, FISTA_d, name)
    __repr__ = _swig_repr

    def __init__(self, L_0=0.1):
        this = _hdim.new_FISTA_d(L_0)
        try:
            self.this.append(this)
        except Exception:
            self.this = this

    def __call__(self, *args):
        return _hdim.FISTA_d___call__(self, *args)
    __swig_destroy__ = _hdim.delete_FISTA_d
    __del__ = lambda self: None
FISTA_d_swigregister = _hdim.FISTA_d_swigregister
FISTA_d_swigregister(FISTA_d)

class CD_f(Solver_f):
    __swig_setmethods__ = {}
    for _s in [Solver_f]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, CD_f, name, value)
    __swig_getmethods__ = {}
    for _s in [Solver_f]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, CD_f, name)
    __repr__ = _swig_repr

    def __init__(self, X, Y, Beta_0):
        this = _hdim.new_CD_f(X, Y, Beta_0)
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _hdim.delete_CD_f
    __del__ = lambda self: None

    def __call__(self, *args):
        return _hdim.CD_f___call__(self, *args)
CD_f_swigregister = _hdim.CD_f_swigregister
CD_f_swigregister(CD_f)

class CD_d(Solver_d):
    __swig_setmethods__ = {}
    for _s in [Solver_d]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, CD_d, name, value)
    __swig_getmethods__ = {}
    for _s in [Solver_d]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, CD_d, name)
    __repr__ = _swig_repr

    def __init__(self, X, Y, Beta_0):
        this = _hdim.new_CD_d(X, Y, Beta_0)
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _hdim.delete_CD_d
    __del__ = lambda self: None

    def __call__(self, *args):
        return _hdim.CD_d___call__(self, *args)
CD_d_swigregister = _hdim.CD_d_swigregister
CD_d_swigregister(CD_d)

# This file is compatible with both classic and new-style classes.


