class cell:
    def __init__(self, prim, xmin, xmax, id, children=[], parent=None, level=0):
        self.prim = prim
        self.id = id
        self.children = children
        self.parent = parent
        self.xmin = xmin
        self.xmax = xmax
        self.x = (xmax + xmin) / 2
        self.dx = xmax-xmin

        self.need_refine = False
        self.need_coarse = False

        self.activating = True

        self.level = level

    def update(self, prim):
        self.prim = prim