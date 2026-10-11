class MiniKernelError(Exception):
    """Base error for user-facing MiniKernel failures."""


class LexError(MiniKernelError):
    pass


class ParseError(MiniKernelError):
    pass


class CheckError(MiniKernelError):
    pass


class CompileError(MiniKernelError):
    pass


class LaunchError(MiniKernelError):
    pass
