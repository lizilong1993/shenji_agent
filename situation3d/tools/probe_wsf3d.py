#!/usr/bin/env python3
"""Inspect the pixel statistics of a WSF3D WMS tile.

Determines whether the raster can be decoded back to a building height, which
is what a 3D building-block rendering would need. Pure standard library: the
container has no pip, no PIL and no numpy.

Usage: python3 probe_wsf3d.py <tile.png> [<tile.png> ...]
"""
import sys
import zlib
import struct
import collections


def read_png(path):
    data = open(path, "rb").read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a png: " + path)
    pos = 8
    idat = b""
    width = height = depth = ctype = None
    palette = None
    trns = None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctag = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctag == b"IHDR":
            width, height, depth, ctype = struct.unpack(">IIBB", chunk[:10])
        elif ctag == b"IDAT":
            idat += chunk
        elif ctag == b"PLTE":
            palette = chunk
        elif ctag == b"tRNS":
            trns = chunk
        elif ctag == b"IEND":
            break
    raw = zlib.decompress(idat)
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    if depth != 8:
        raise ValueError("only 8-bit supported, got %s" % depth)
    stride = width * channels
    out = bytearray(height * stride)
    prev = bytearray(stride)
    p = 0
    for y in range(height):
        f = raw[p]
        p += 1
        line = bytearray(raw[p:p + stride])
        p += stride
        if f == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif f == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif f == 3:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif f == 4:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return width, height, channels, ctype, bytes(out), palette, trns


def main(paths):
    for path in paths:
        w, h, ch, ctype, pix, palette, trns = read_png(path)
        print("== %s" % path)
        print("   %dx%d channels=%d colortype=%d palette=%s"
              % (w, h, ch, ctype, "yes" if palette else "no"))
        hist = collections.Counter()
        alpha_hist = collections.Counter()
        n = w * h
        for i in range(n):
            off = i * ch
            if ctype == 3:
                idx = pix[off]
                r, g, b = palette[idx * 3:idx * 3 + 3]
            elif ch >= 3:
                r, g, b = pix[off], pix[off + 1], pix[off + 2]
            else:
                r = g = b = pix[off]
            a = pix[off + ch - 1] if (ctype in (4, 6)) else 255
            hist[(r, g, b)] += 1
            alpha_hist[a] += 1
        print("   distinct colours: %d" % len(hist))
        for (r, g, b), c in hist.most_common(8):
            print("     rgb(%3d,%3d,%3d) x%-7d %5.1f%%" % (r, g, b, c, 100.0 * c / n))
        print("   alpha values: %s" % dict(alpha_hist.most_common(6)))
        # Is red == green == blue everywhere (i.e. a greyscale height ramp)?
        grey = sum(c for (r, g, b), c in hist.items() if r == g == b)
        print("   greyscale pixels: %.1f%%" % (100.0 * grey / n))


if __name__ == "__main__":
    main(sys.argv[1:])
