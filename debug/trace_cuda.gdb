# Trace every exported CUDA driver entry point (cu[A-Z]...).
#
# The rbreak must happen *after* libcuda.so.1 is mapped, so stop at cuInit
# first.  Log goes to /tmp/cuda_calls.log.
#
#   gdb -batch -x debug/trace_cuda.gdb --args python your_script.py
#
set pagination off
set confirm off
set breakpoint pending on
break cuInit
run
python
import gdb
out = open('/tmp/cuda_calls.log', 'w', buffering=1)

try:
    gdb.execute('rbreak ^cu[A-Z]', to_string=True)
except gdb.error as e:
    out.write('rbreak failed: %s\n' % e)

class Entry(gdb.Breakpoint):
    def __init__(self, spec):
        super().__init__(spec, gdb.BP_BREAKPOINT, internal=True)
        self.silent = True
    def stop(self):
        try:
            out.write(self.location + '\n')
        except Exception as e:
            out.write('ERR %s\n' % e)
        return False

for bp in list(gdb.breakpoints()):
    if bp.is_valid():
        try:
            loc = bp.location
        except Exception:
            continue
        try:
            Entry(loc)
        except Exception as e:
            out.write('create failed %s: %s\n' % (loc, e))
        bp.delete()
end
continue
