def calculate_discount(price, discount_percent):
    # Bug: Division by zero if price is 0? No, discount can be > 100? 
    # Let's use a clear bug that static analysis catches easily.
    # Unused variable and a logical error.
    final_price = price - (price * (discount_percent / 100))
    
    # Bug: Using variable before assignment or out of scope, or mutable default
    return final_price

def append_to_list(value, my_list=[]):
    # Bug: Mutable default argument (anti-pattern)
    my_list.append(value)
    
    # Bug: loop with while True without break
    # while True:
    #     pass
    
    return my_list
