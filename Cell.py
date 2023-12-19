class Cell:
    """
    Объект "ячейка"
    Каждая ячейка имеет координату, цвет, таргет, расположенный объект и соседние ячейки.
    Ячейка имеет 4 соседних ячеек: передний(front), задний(back), левый(left) и правый(right) 
    """

    def __init__(self, coordinate, color='w', target='0', located=None, 
                 front=None, back=None, left=None, right=None ) -> None:
        self.coordinate = coordinate
        self.color = color
        self.target = target
        self.located = located

        self.front = front
        self.back = back
        self.left = left
        self.right = right
    def __repr__(self) -> str:
        return "*{},{:2},{},{}*".format(
            self.coordinate, self.color, self.target, self.located)
    
    @property
    def adj(self):
         return [cell for cell in [self.front, self.back, self.left, self.right] if cell]

    def  __eq__(self, cell):
        if self.coordinate == cell.coordinate:
            return True
        else: 
            return False
    
    def set_adj(self, front=None, back=None, left=None, right=None ):
        self.front = front
        self.back = back
        self.left = left
        self.right = right
        return True