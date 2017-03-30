from threading import Thread
from multiprocessing import Queue, Process, Lock


class LoopWork(Thread):
    def run(self):
        sum = 0
        while True:
            for i in range(5000000):
                sum += i
        print(sum)


def self_sum(q, k, l, o):
    while True:
        n = q.get()
        o.put(n)
        l.acquire()
        print(n)
        l.release()


def main():
    q = Queue()
    o = Queue(maxsize=4)
    l = Lock()
    links = [1, 2, 3, 4]
    ps = []
    for id in links:
        p = Process(target=self_sum, args=(q, id, l, o))
        ps.append(p)
        p.daemon = True
        p.start()
    for i in range(1000):
        q.put(i)
    # for p in ps:
    #    p.join()
    c = 0
    while True:
        o.get()
        c += 1
        if c == 1000:
            break

    ###########################
    # ts = []                 #
    # for x in range(4):      #
    #     worker = LoopWork() #
    #     ts.append(worker)   #
    #     worker.start()      #
    # for t in ts:            #
    #     t.join()            #
    ###########################


if __name__ == '__main__':
    main()
