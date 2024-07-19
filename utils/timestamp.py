import datetime

class ColorPalette:
    OK = '\033[1;92m'  # GREEN
    WARNING = '\033[1;93m'  # YELLOW
    FAIL = '\033[1;91m'  # RED
    INFO = '\033[1;94m'  # BLUE
    RESET = '\033[0m'  # RESET COLOR

def print_with_timestamp(is_master):
    """
    This function adds a timestamp to each log info and supports color based on the input mode.
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        if is_master:
            timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
            mode = kwargs.pop("mode", "OK")

            if mode:
                color = getattr(ColorPalette, mode, ColorPalette.RESET)
                args = (f"{color} {timestamp}",) + (ColorPalette.RESET,) + args
            else:
                args = (f"{timestamp} ",) + args

            builtin_print(*args, **kwargs)

    __builtin__.print = print