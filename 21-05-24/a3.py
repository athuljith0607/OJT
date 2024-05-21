class IncrementingIterator:
    def _init_(self, start=0):
        self.current = start
    
    def _iter_(self):
        return self
    
    def _next_(self):
        current_value = self.current
        self.current += 1
        return current_value

iterator = IncrementingIterator()


print(next(iterator))
print(next(iterator))  
print(next(iterator))