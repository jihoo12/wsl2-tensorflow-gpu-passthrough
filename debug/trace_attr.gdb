# Find which CU_DEVICE_ATTRIBUTE query fails.
#
# Logs every cuDeviceGetAttribute(pi, attrib, dev) call ($esi = attrib,
# $edx = dev on x86-64 SysV) and dumps a backtrace whenever cuGetErrorName()
# is called -- that is TensorFlow converting a CUresult into a Status.
#
#   gdb -batch -x debug/trace_attr.gdb --args python your_script.py
#
# Log goes to /tmp/attr_calls.log.
#
set pagination off
set confirm off
set breakpoint pending on
break cuInit
run
python
import gdb
out = open('/tmp/attr_calls.log', 'w', buffering=1)

class AttrBP(gdb.Breakpoint):
    def stop(self):
        try:
            attr = int(gdb.parse_and_eval('$esi')) & 0xffffffff
            dev = int(gdb.parse_and_eval('$edx')) & 0xffffffff
            out.write('attr %d dev %d\n' % (attr, dev))
        except Exception as e:
            out.write('attr ERR %s\n' % e)
        return False

class ErrBP(gdb.Breakpoint):
    def stop(self):
        try:
            code = int(gdb.parse_and_eval('$edi')) & 0xffffffff
            out.write('cuGetErrorName code %d\n' % code)
        except Exception as e:
            out.write('err ERR %s\n' % e)
        out.write('--- backtrace ---\n')
        try:
            for line in gdb.execute('bt 12', to_string=True).splitlines():
                out.write(line + '\n')
        except Exception as e:
            out.write('bt ERR %s\n' % e)
        return False

for spec in ('cuDeviceGetAttribute', 'cuGetErrorName'):
    gdb.execute('break %s' % spec, to_string=True)

for bp in list(gdb.breakpoints()):
    try:
        loc = bp.location
    except Exception:
        continue
    if loc == 'cuInit':
        bp.delete()
        continue
    if loc in ('cuDeviceGetAttribute', 'cuDeviceGetAttribute@plt'):
        if loc == 'cuDeviceGetAttribute':
            AttrBP(loc)
        bp.delete()
    elif loc == 'cuGetErrorName':
        ErrBP(loc)
        bp.delete()
    else:
        bp.delete()
end
continue
