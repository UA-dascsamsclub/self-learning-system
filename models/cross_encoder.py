from sentence_transformers import CrossEncoder
model_path = "models/model_ce"
model = CrossEncoder(model_path, num_labels=4, automodel_args={'ignore_mismatched_sizes': True})

'''
#Testing the model
scores = model.predict([('carrots', 'Fresh-Cut Vegetable Tray and Ranch Dressing, priced per pound'), 
                        ('carrots', 'Sweet Potato & Carrot Recipe, Grain Free Dry Dog Food , 23.5 lbs.'), 
                        ('carrots', 'Carrot Bar Cake, 38 oz.'),
                        ('carrots', 'Whole Carrots, 5 lbs.'),
                        ('carrots', 'Blue Buffalo Nudges Homestyle Chicken, Peas, and Carrots Natural Dog Treats, 40 oz.')])
print(scores)
'''