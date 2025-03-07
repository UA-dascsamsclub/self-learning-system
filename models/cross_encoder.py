from sentence_transformers import CrossEncoder
model = CrossEncoder('/Users/sarahlawlis/Desktop/repos/self-learning-system/models/model_ce')

scores = model.predict([('carrots', 'Fresh-Cut Vegetable Tray and Ranch Dressing, priced per pound'), 
                        ('carrots', 'Sweet Potato & Carrot Recipe, Grain Free Dry Dog Food , 23.5 lbs.'), 
                        ('carrots', 'Carrot Bar Cake, 38 oz.'),
                        ('carrots', 'Whole Carrots, 5 lbs.'),
                        ('carrots', 'Blue Buffalo Nudges Homestyle Chicken, Peas, and Carrots Natural Dog Treats, 40 oz.')])
print(scores)