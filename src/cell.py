class cell:
    def __init__(self, prim, xmin, xmax, id, children=None, parent=None, level=0):
        self.prim = prim
        self.id = id
        if children is None:
            self.children = []
        else:
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

    def __repr__(self ):
        return(
            f"Cell("
            f"prim={self.prim}, "
            f"xmin={self.xmin:.3f}, "
            f"xmax={self.xmax:.3f}, "
            f"id={self.id}, "
            f"children={self.children}, "
            f"parent={self.parent}, "
            f"level={self.level}, "
            f"need_refine={self.need_refine}, "
            f"need_coarse={self.need_coarse}, "
            f"activating={self.activating})"
        )