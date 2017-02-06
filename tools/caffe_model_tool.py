

def make_pool(args):
    pass


class God:
    keywords = ['C',
                'P',
                'D',
                'B',
                'U',
                'A',
                'P',
                'R']

    def __init__(self):
        self._conv_count = 0
        self._pool_count = 0
        self._drop_count = 0
        self._batchnorm_count = 0
        self._add_count = 0
        self._upsample_count = 0
        self._relu_count = 0


class Machine:

    def make_conv(self, line):
        cc = line[self._cursor]
        if cc.isdigit():
            pass
        else:
            return self._conv()

    def _conv(kern_size=3, stride=2, pad=1):

        pass

    def __init__(self):
        self.ctx = God()
        self._cursor = 0

    def be_angry(self):
        raise Exception("I'm angry")
        print "hehe"

    def eat(self, line):
        self._cursor = 0
        self._scan_line(line.upper())
        pass

    def _parse_block(self, block, line):
        block = self.make(block)
        pass

    def make(self, block, line):
        if block == 'C':
            return self.make_conv(line)
        pass

    def tick(self, c=1):
        self._cursor += c

    def _scan_line(self, line):
        current_block = line[self._cursor]
        if current_block in God.keywords:
            self.tick()
            self._parse_block(current_block, line)
        else:
            self.be_angry()


def load(file_):
    ret = []
    for line in open(file_):
        ret.append(line)
    return ret


def main(file_path):
    lines = load(file_path)
    mac = Machine()
    for line in lines:
        mac.eat(line)


if __name__ == '__main__':
    import sys
    # main(sys.argv[1])
    mac = Machine()
    mac.eat("ii")
