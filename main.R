# Importowanie pakietów z opcjonalną instalacją
required_packages <- c("ScottKnottESD", "readr", "ggplot2", "gridExtra", "tidyverse", 
                       "psych", "FSA", "lattice", "coin", "PMCMRplus", "rcompanion", "DescTools")

for (package in required_packages) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package)
    library(package, character.only = TRUE)
  } else {
    library(package, character.only = TRUE)
  }
}


# Lista nazw folderów
folder_names <- c("breast-cancer", "nba", "pistachio", "sonar", "weather")

# Mapowanie klasyfikatorów
classifier_mapping <- c(
  "matthews_LogisticRegression.csv" = "LR",
  "matthews_SVC.csv" = "SVC",
  "matthews_DecisionTreeClassifier.csv" = "DT",
  "matthews_RandomForestClassifier.csv" = "RF",
  "matthews_KNeighborsClassifier.csv" = "KNN",
  "matthews_GaussianNB.csv" = "NB",
  "matthews_XGBClassifier.csv" = "XGB",
  "matthews_MLPClassifier.csv" = "MLP"
)

# Inicjalizacja ramki danych do przechowywania danych
combined_data <- data.frame()
kendall_results <- data.frame()


# Pętla po każdym folderze
for (folder_name in folder_names) {
  
  # Lista plików w folderze
  files <- list.files(path = folder_name, pattern = "\\.csv", full.names = TRUE)
  
  # Inicjalizacja wartości min i max
  global_min <- Inf
  global_max <- -Inf
  
  # Inicjalizacja pustej listy do przechowywania wykresów
  plots <- list()
  
  # Inicjalizacja ramki danych do przechowywania wartości MCC do analizy Kendall W
  kendall_data <- data.frame()
  
  # Pętla po każdym pliku w folderze
  for (i in 1:length(files)) {
    # Wczytanie danych
    model_performance <- read_csv(files[i], col_types = cols(.default = "d"))
    
    # Nadanie nazw kolumn
    colnames(model_performance) <- c("PCA", "LDA", "CA", "Chi2", "RFE", "VT", "Brak")
    
    # Konwersja ciągów znaków na liczby zmiennoprzecinkowe
    model_performance <- as.matrix(model_performance)
    model_performance <- apply(model_performance, c(1, 2), as.numeric)
    
    # Aktualizacja wartości min i max
    local_min <- min(model_performance)
    local_max <- max(model_performance)
    
    if (local_min < global_min) {
      global_min <- local_min
    }
    
    if (local_max > global_max) {
      global_max <- local_max
    }
    
    # Przygotowanie danych i analiza ScottKnottESD
    sk_results <- sk_esd(model_performance, version="np")
    sk_ranks <- data.frame(model = names(sk_results$groups),
                           rank = paste0(sk_results$groups))
    
    # Przygotowanie ramki danych do generowania wizualizacji
    plot_data <- melt(model_performance)
    plot_data <- merge(plot_data, sk_ranks, by.x = 'Var2', by.y = 'model')
    
    # Generowanie wizualizacji z niestandardowym zakresem osi Y i swobodnym rozstawem osi X
    methods_colors <- c("#787878", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#003BB2")
    names(methods_colors) <- levels(plot_data$Var2)
    
    if (i %% 2 == 0) {
      ylab_text <- NULL
    } else {
      ylab_text <- 'MCC'
    }
    
    classifier_name <- classifier_mapping[basename(files[i])]
    
    boxplot_mcc <- ggplot(data = plot_data, aes(x = Var2, y = value, fill = Var2)) +
      geom_boxplot(outlier.shape = 16, outlier.size = 1, notch = FALSE, alpha = 0.7, fatten = 1) +
      ylim(global_min, global_max) +
      facet_grid(~rank, scales = 'free_x', space = "free_x") +
      scale_fill_manual(values = methods_colors) +
      ylab(ylab_text) + xlab(NULL) +
      ggtitle(classifier_name) +
      theme_bw() +
      theme(plot.title = element_text(size = 9, hjust = 0.5),
            axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
            text = element_text(size = 8),
            legend.position = 'none',
            panel.spacing = unit(1, "lines"),
            panel.border = element_rect(colour = "grey50", fill = NA, size = 1),
            axis.title.x = element_text(size = 8),
            axis.title.y = element_text(size = 8))
    
    plots[[i]] <- boxplot_mcc
    
    mcc_values <- data.frame(iteration_id = 1:100, MCC = model_performance[1:100,])
    mcc_values$method <- classifier_name
    
    kendall_data <- rbind(kendall_data, mcc_values)
    
    plot_data$dataset <- folder_name
    plot_data$classifier <- classifier_name
    
    combined_data <- rbind(combined_data, plot_data)
  }
  
  # Zapisywanie wykresu do pliku PDF
  ggsave(filename = paste0(folder_name, "_output.pdf"), 
         plot = grid.arrange(grobs = plots, ncol = 2, nrow = 4), 
         width = 8.27, height = 11.69)
  
  # Analiza Kendall W dla różnych klasyfikatorów
  for (classifier_name in unique(kendall_data$method)) {
    
    # Filtracja danych dla danego klasyfikatora
    classifier_data <- kendall_data %>% filter(method == classifier_name) %>% select(-iteration_id, -method)
    
    # Sprawdzenie, czy dane dla klasyfikatora istnieją
    if (nrow(classifier_data) > 0) {
      
      # Obliczenie statystyki Kendall W
      KendallW.MCC <- KendallW(classifier_data, correct = TRUE, test = TRUE)
      
      # Utworzenie wiersza z wynikami
      result_row <- data.frame(
        Folder = folder_name,
        Classifier = classifier_name,
        KendallW_Statistic = KendallW.MCC$estimate,
        KendallW_P_Value = KendallW.MCC$p.value
      )
      
      # Dodanie wyników do zbiorczych wyników
      kendall_results <- rbind(kendall_results, result_row)
    }
  }
  
  # Przetworzenie wyników do postaci tabeli
  kendall_cross_table <- kendall_results %>%
    spread(Classifier, KendallW_Statistic) %>%
    select(Folder, KNN, XGB, DT, LR, NB, MLP, RF, SVC)
  
  # Przekształcenie tabeli do długiego formatu
  kendall_cross_table_long <- kendall_cross_table %>%
    pivot_longer(cols = -Folder, names_to = "Classifier", values_to = "KendallW_Statistic") %>%
    filter(!is.na(KendallW_Statistic))
  
  # Przekształcenie tabeli z powrotem do szerokiego formatu
  kendall_cross_table_wide <- kendall_cross_table_long %>%
    pivot_wider(names_from = Classifier, values_from = KendallW_Statistic)
  
  # Zmiana nazw kolumn
  kendall_cross_table_wide <- kendall_cross_table_wide %>%
    rename(dataset = Folder)
  
  # Ustawienie nazw wierszy
  kendall_cross_table_wide <- column_to_rownames(kendall_cross_table_wide, var = "dataset")
  
  # Dodanie wiersza z wartościami średnimi
  kendall_cross_table_wide[nrow(kendall_cross_table_wide) + 1,] <- colMeans(kendall_cross_table_wide, na.rm = TRUE)
  row.names(kendall_cross_table_wide)[row.names(kendall_cross_table_wide) == "6"] <- "Średnia"
  
  # Sortowanie wyników względem średniej
  kendall_cross_table_wide <- kendall_cross_table_wide[, order(as.numeric(kendall_cross_table_wide["Średnia", ]), decreasing = TRUE)]
  
  # Zaokrąglenie wartości w tabeli
  kendall_cross_table_wide[] <- lapply(kendall_cross_table_wide, function(x) round(x, 3))
  
  # Zapisanie tabeli do pliku CSV
  write.csv(kendall_cross_table_wide, "kendallW_summary.csv")
}

# Konwersja kolumny rank na typ numeryczny
combined_data$rank <- as.numeric(combined_data$rank)

# Definicja kolorów dla różnych metod
methods_colors <- c("#787878", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#003BB2")
names(methods_colors) <- levels(combined_data$Var2)

# Tworzenie wykresu pudełkowego
boxplot_overall <- ggplot(data = combined_data %>% 
                            arrange(Var2, rank) %>% 
                            group_by(Var2) %>% 
                            mutate(mean_rank = mean(rank)),
                          aes(x = reorder(Var2, mean_rank), y = rank, fill = Var2)) +
  geom_boxplot(outlier.shape = 16, outlier.size = 1, notch = FALSE, alpha = 0.7, fatten = 1) +
  stat_summary(
    fun = mean,
    geom = "crossbar",
    position = position_dodge(width = 0.8),
    width = 0.7,
    fatten = 1,
    col = "black",
    linetype = "dashed"
  ) +
  scale_fill_manual(values = methods_colors) +
  ylab("Ranga") + xlab(NULL) +
  theme_bw() +
  scale_y_reverse() +
  theme(
    plot.title = element_text(size = 14, hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    text = element_text(size = 10),
    legend.position = 'none',
    panel.spacing = unit(1, "lines"),
    panel.border = element_rect(colour = "grey50", fill = NA, size = 1),
    axis.title.x = element_text(size = 10),
    axis.title.y = element_text(size = 10)
  )

# Zapisanie wykresu do pliku SVG
ggsave("boxplot.svg", plot = boxplot_overall, width = 8, height = 6, units = "in")

# Funkcja do obliczania mediany z zachowaniem typu danych
median_with_type <- function(x) {
  if (is.numeric(x)) {
    return(median(x, na.rm = TRUE))
  } else {
    return(as.character("Mediana"))
  }
}


# Tworzenie tabeli dla klasyfikatora
pivot_table_classifier <- combined_data %>%
  select(classifier, Var2, rank) %>%
  mutate(rank = as.numeric(rank)) %>%
  group_by(classifier, Var2) %>%
  summarize(mean_rank = mean(rank), .groups = 'keep') %>%
  spread(key = Var2, value = mean_rank)

pivot_table_classifier <- pivot_table_classifier %>%
  column_to_rownames(var = "classifier")

average_row_classifier <- pivot_table_classifier %>%
  summarize_all(mean) %>%
  mutate(classifier = "Średnia") %>%
  column_to_rownames(var = "classifier")

pivot_table_classifier <- bind_rows(pivot_table_classifier, average_row_classifier)

# Tworzenie tabeli dla zbioru danych
pivot_table_dataset <- combined_data %>%
  select(dataset, Var2, rank) %>%
  mutate(rank = as.numeric(rank)) %>%
  group_by(dataset, Var2) %>%
  summarize(mean_rank = mean(rank), .groups = 'keep') %>%
  spread(key = Var2, value = mean_rank)

pivot_table_dataset <- pivot_table_dataset %>%
  column_to_rownames(var = "dataset")

average_row_dataset <- pivot_table_dataset %>%
  summarize_all(mean) %>%
  mutate(dataset = "Średnia") %>%
  column_to_rownames(var = "dataset")

pivot_table_dataset <- bind_rows(pivot_table_dataset, average_row_dataset)

# Zaokrąglenie wartości w tabelach
pivot_table_classifier[] <- lapply(pivot_table_classifier, function(x) round(x, 2))
pivot_table_dataset[] <- lapply(pivot_table_dataset, function(x) round(x, 2))

# Transponowanie ramki danych
transposed_df <- as.data.frame(t(pivot_table_classifier))

# Konwersja wiersza "Średnia" na typ numeryczny
transposed_df$Średnia <- as.numeric(transposed_df$Średnia)

# Sortowanie ramki danych względem kolumny "Średnia"
sorted_df <- transposed_df[order(transposed_df$Średnia), ]

# Transponowanie posortowanej ramki danych z powrotem do pierwotnego formatu
pivot_table_classifier_sorted <- as.data.frame(t(sorted_df))

# Transponowanie ramki danych
transposed_df2 <- as.data.frame(t(pivot_table_dataset))

# Konwersja wiersza "Średnia" na typ numeryczny
transposed_df2$Średnia <- as.numeric(transposed_df$Średnia)

# Sortowanie ramki danych względem kolumny "Średnia"
sorted_df2 <- transposed_df2[order(transposed_df$Średnia), ]

# Transponowanie posortowanej ramki danych z powrotem do pierwotnego formatu
pivot_table_dataset_sorted <- as.data.frame(t(sorted_df2))

# ----- SORTOWANIE 
pivot_table_dataset_sorted <- pivot_table_dataset_sorted[, order(as.numeric(pivot_table_dataset_sorted["Średnia", ]), decreasing = FALSE)]
pivot_table_classifier_sorted <- pivot_table_classifier_sorted[, order(as.numeric(pivot_table_classifier_sorted["Średnia", ]), decreasing = FALSE)]
#-----------------

# Obliczanie median dla każdej metody, każdego zbioru danych i każdego klasyfikatora
medians_per_method <- combined_data %>%
  group_by(Var2) %>%
  summarise(Mediana = median(rank, na.rm = TRUE))

medians_per_dataset <- combined_data %>%
  group_by(dataset) %>%
  summarise(Mediana = median(rank, na.rm = TRUE))

medians_per_classifier <- combined_data %>%
  group_by(classifier) %>%
  summarise(Mediana = median(rank, na.rm = TRUE))

# Konwertowanie wyników na tabelę
medians_per_method <- column_to_rownames(medians_per_method, var = "Var2")
medians_per_dataset <- column_to_rownames(medians_per_dataset, var = "dataset")
medians_per_classifier <- column_to_rownames(medians_per_classifier, var = "classifier")

# Transponowanie ramki danych medians_per_method
medians_per_method_transposed <- t(medians_per_method)

# Ustawianie nazw wierszy na "Mediana"
rownames(medians_per_method_transposed) <- "Mediana"

# Dodawanie median do pivot_table_dataset_sorted
pivot_table_dataset_sorted <- rbind(pivot_table_dataset_sorted[1:(nrow(pivot_table_dataset_sorted)-1), ],  # wszystko oprócz ostatniego wiersza
                                    medians_per_method_transposed,
                                    pivot_table_dataset_sorted[nrow(pivot_table_dataset_sorted), ])  # ostatni wiersz






# # Przekształć ramkę danych medians_per_classifier
# medians_per_classifier["Średnia", ] <- "-"

# 
# # Ustaw nazwy kolumn na "Mediana"
# colnames(medians_per_classifier) <- "Mediana"
# 
# 
# # Dodaj mediany do pivot_table_classifier_sorted
# pivot_table_classifier_sorted <- cbind(pivot_table_classifier_sorted, medians_per_classifier)


