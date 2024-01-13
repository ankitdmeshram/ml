from collections import Counter

transactions = [[11, 12, 13], [12, 13, 14], [14, 15], [11, 12, 14], [11, 12, 13, 15], [11, 12, 13, 14]]

min_support = 2  # Minimum support count

# Count item frequencies
item_counts = Counter(item for sublist in transactions for item in sublist)

# Filter frequent items
frequent_items = {item for item, count in item_counts.items() if count >= min_support}

# Generate association rules
rules = []
for item1 in frequent_items:
   for item2 in frequent_items:
       if item1 != item2:
           support12 = sum(item1 in transaction and item2 in transaction for transaction in transactions)
           support1 = sum(item1 in transaction for transaction in transactions)
           confidence = support12 / support1
           if confidence >= 0.5:  # Minimum confidence threshold
               rules.append((item1, item2, confidence))

# Print results
print("Association Rules:")
for rule in rules:
   antecedent, consequent, confidence = rule
   print(f"{antecedent} -> {consequent} (confidence: {confidence:.2f})")
