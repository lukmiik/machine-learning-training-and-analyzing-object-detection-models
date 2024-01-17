install.packages("ggplot2")
library(ggplot2)

csv_file_path <- "path/to/csv/file"

data <- read.csv(csv_file_path)

View(data)

library(tidyr)
data_long <- gather(data, key = "Class", value = "mAP", -Model)

ggplot(data_long, aes(x = Class, y = mAP, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), color = "black", width = 0.6) +
  labs(title = "Comparison of mAP Across Models for Each Class in <dataset name>
dataset",
       x = "Object Detection Classes",
       y = "Mean Average Precision (mAP)",
       fill = "Object Detection Models") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
