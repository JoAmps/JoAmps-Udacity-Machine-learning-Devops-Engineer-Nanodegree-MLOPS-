# Model Card
Original paper introducing the concept of model card canbe found here: https://arxiv.org/pdf/1810.03993.pdf
## Model Details
Hyacinth K. Ampadu created the model. It is Random forest using the default hyperparameters in scikit-learn 0.24.2.
## Intended Use
This model should be used to predict if individuals make income above 50k dollars a year or below 50K dollars per year based off a handful of attributes.
## Training Data
The training data comprised of 80% of the original data. The target class the salary, in 2 categories, salaries over 50K dollars, and salaries below 50K dollars. Training data was one hot encoded and label binarized.
## Evaluation Data
Evaluation data comprised same parameters as training data, having the remaining 20% of the original data. It was one hot encoded, but no label binarization done.
## Metrics
The model was evaluated using F1 score, precision and recall. The value of precision was 0.7454, recall was 0.615 and f1 score was 0.6739
## Ethical Considerations
The Dataset has data containing race and gender, which could potentially discriminate against individuals in such brackets, hence a more deep dive into this may be neccesary
## Caveats and Recommendations
Given some countries have much more data than other countries, more work needs to be done to capture more of such underrepresented countries to result in a more fair model