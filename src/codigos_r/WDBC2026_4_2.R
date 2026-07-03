###WDBC2026_4_2

library(tidymodels)
library(tidyverse)
library(tidyclust)
library(factoextra)
library(FactoMineR)
library(cluster)
library(mlr)
library(GGally)
library(ClusterR)
library(vegan)
library(NbClust)
library(gridExtra)
library(grid)
library(lattice)
require(igraph)
library(recipes)
library(rsample)
library(workflows)
library(tune)
library(parameters)
library(pvclust)
library(hopkins)
library(flextable)
library(plotly)

df<-read.csv("/home/yamorza/Documentos/espacio_trabajo_intro_analisis_dat2/diplomado_sessions/Modulo_6/sesion_3/habits.csv")

###Variable de clasificacion: diagnosis

df <- df %>%
  mutate(
    exam_quantile = ntile(exam_score, 4),
    exam_quantile = factor(exam_quantile, 
                           labels = c("Q1 (Lowest)", "Q2(middle_1)", "Q3 (middle_2)","Q4 (highest)"))
  )

df %>% dim()

df %>% glimpse()

df %>% head()

df %>% count(exam_quantile)

###MÉTODOS NO JERÁRQUICOS
###K-MEANS

###Explorando (que es gerundio) el número de grupos (K)

###Basándonos en SSE

rec_df <- recipe(~.,data = df) %>%
  update_role(exam_quantile, new_role = "id") %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) 

rec_df

kmeans_spec <- k_means(num_clusters = tune())

kmeans_wf <- workflow(rec_df, kmeans_spec)

kmeans_wf <- kmeans_wf %>% 
  update_model(kmeans_spec)

grid <- tibble(num_clusters = 1:10)

set.seed(123)
boots <- bootstraps(df, times = 10)

res <- tune_cluster(
  kmeans_wf,
  resamples = boots,
  grid = grid,
  metrics = cluster_metric_set(sse_within_total, sse_total, sse_ratio)
)

res_metrics <- collect_metrics(res)%>% print(n=Inf)

best <- res %>%
  select_best(metric="sse_ratio")  ###Observese que este criterio es un poco "chafa", ya que esta medida
best                               ### en general va decreciendo con el aumento en el valor de K

###La gráfica de codo

res_metrics %>%
  filter(.metric == "sse_ratio") %>%
  ggplot(aes(x = num_clusters, y = mean)) +
  geom_point(col="darkblue",size=2) +
  geom_line(col="darkred") +
  theme_minimal() +
  ylab("mean WSS/TSS ratio") +
  xlab("Número de clusters") +
  scale_x_continuous(breaks = 1:10)+
  ggtitle("WDBC: Gráfica de codo K-MEANS")

###Validacion cruzada

df_cv <- vfold_cv(df, v = 10)

clust_num_grid <- grid_regular(num_clusters(),levels = 10)

clust_num_grid

res1 <- tune_cluster(
  kmeans_wf,
  resamples = df_cv,
  grid = clust_num_grid,
  control = control_grid(save_pred = TRUE, extract = identity),
  metrics = cluster_metric_set(sse_within_total, sse_total, sse_ratio)
)


res1_metrics <- res1 %>% collect_metrics()%>% print(n=Inf)


best1 <- res1 %>%
  select_best(metric="sse_ratio")  ###Misma observación que antes
best1


res1_metrics %>%
  filter(.metric == "sse_ratio") %>%
  ggplot(aes(x = num_clusters, y = mean)) +
  geom_point(col="darkblue",size=2) +
  geom_line(col="red") +
  theme_minimal() +
  ylab("mean WSS/TSS ratio cv") +
  xlab("Number of clusters") +
  scale_x_continuous(breaks = 1:10)+
  ggtitle("WDBC: Gráfica de codo K-MEANS con CV")

###A traves del metodo de silueta (silhouette)

cluster_grid <- tibble(num_clusters = 2:10)

kmeans_spec <- k_means(num_clusters = tune())

set.seed(123)
tune_results <- tune_cluster(
 kmeans_spec,
  ~ .,
  data = scale(df[,-1]),
  resamples = vfold_cv(df[,-1], v = 10),
  grid = cluster_grid,
  metrics = cluster_metric_set(silhouette_avg)
)

show_best(tune_results, metric = "silhouette_avg")

best2 <- tune_results %>%
  select_best(metric="silhouette_avg")  
best2


metrics_df <- collect_metrics(tune_results)

ggplot(metrics_df, aes(x = num_clusters, y = mean)) +
  geom_line(color = "steelblue", size = 1) +
  geom_point(color = "darkred", size = 3) +
  labs(
    title = "Proceso tuning de cluster: Silhouette Method",
    subtitle = "Evaluación de k = 1 to 10 con 10 folds",
    x = "Número de of Clusters (k)",
    y = "Promedio Silhouette Score"
  ) +
  theme_minimal()



###La silhouette muestra que hay DOS GRUPOS subyacentes a estos datos

###K-MEANS con 4 CLUSTERS

recipe_df <- recipe(~.,data = df) %>%
  update_role(exam_quantile, new_role = "id") %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) 

recipe_df

kmeans_df <- k_means(num_clusters = 4)

kmeans_df_wf <- workflow(recipe_df, kmeans_df)

kmeans_df_wf <- kmeans_df_wf %>% 
  update_model(kmeans_df)

kmeans_df_fit<-fit(kmeans_df_wf, data = df)
kmeans_df_fit

kmeans_summary <- kmeans_df_fit |>
  extract_fit_summary()

kmeans_summary |> str()

kmeans_df_fit |>
  extract_cluster_assignment()


tibble(
  orig_labels = kmeans_summary$orig_labels,
  standard_labels = kmeans_summary$cluster_assignments
)

kmeans_df_fit |>
  extract_centroids()

###Suma de cuadrados del error
###Una métrica sencilla es la suma cuadrados de los errores dentro de los conglomerados (WSS, por sus siglas en inglés), 
###que mide la suma de todas las distancias desde las observaciones hasta el centro de su respectivo conglomerado. 
###A veces, este valor se normaliza con respecto a la suma total de los errores al cuadrado (TSS), que representa la 
###distancia de todas las observaciones al centroide global; en particular, suele calcularse la relación WSS/TSS. 
###En principio, valores bajos de WSS o de la relación WSS/TSS sugieren que las observaciones dentro de los conglomerados 
###están más cerca (son más similares) entre sí que respecto a los otros conglomerados.

kmeans_summary$sse_within_total_total
kmeans_summary$sse_total

kmeans_df_fit |> sse_within_total()
kmeans_df_fit |> sse_total()

kmeans_df_fit |> sse_ratio()

kmeans_df_fit |>
  sse_within()

kmeans_df_fit |>
  silhouette_avg(df)

###Predicción

set.seed(123)
predict(kmeans_df_fit, new_data = slice_sample(df, n = 10))

###Gráfica

clustered_data <- augment(kmeans_df_fit, new_data = df)

centroids <- tidy(kmeans_df_fit)

ggplot() +
  geom_point(data = clustered_data, 
             aes(x=scale(study_hours_per_day), y=scale(exam_score), color = .pred_cluster), 
             alpha = 0.6) +
  geom_point(data = centroids, 
             aes(x=scale(study_hours_per_day), y=scale(exam_score)), 
             color = "darkmagenta", size = 3, shape = 3, stroke = 2) +
  theme_minimal()+labs(
    title = "WDBC: K-MEANS agrupación",
    subtitle = "Número de grupos: k = 2",
    x = "radius_mean",
    y = "texture_mean"
  )


###Explorando el numero de clustes K subyacentes a estos datos con otros metodos

df1 <- df %>% 
  dplyr::select(-exam_quantile) %>% 
  dplyr::select(where(is.numeric)) %>% 
  scale()

fviz_nbclust(df1, kmeans, method = "wss")+labs(x ="Número de clusters")+labs(y="Total suma de cuadrados intra clusters")+labs(title = "Número óptimo de clusters")

opt<-Optimal_Clusters_KMeans(df1, max_clusters=10,plot_clusters = TRUE,criterion="WCSSE")

fviz_nbclust(df1, kmeans, method = "silhouette")+labs(x ="Número de clusters")+labs(y="Promedio de silueta")+labs(title = "Número óptimo de clusters")

opt1<-Optimal_Clusters_KMeans(df1, max_clusters=10, plot_clusters = TRUE, criterion="silhouette")

opt2<-Optimal_Clusters_KMeans(df1, max_clusters=10, plot_clusters = TRUE, criterion = "variance_explained",fK_threshold = 0.90)

fviz_nbclust(df1, kmeans, method = "gap_stat")+labs(x ="Número de clusters")+labs(y="GAP")+labs(title = "Número óptimo de clusters")

fit <- cascadeKM(df1, 2, 10, iter = 500)
plot(fit, sortg = TRUE, grpmts.plot = TRUE)

opt_aic<-Optimal_Clusters_KMeans(df1, 10, 'euclidean', plot_clusters=TRUE,criterion="AIC")

nb <- NbClust(df1, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans", index ="all")

names(nb) 

nb$Best.nc

###Cuncluimos que hay K=2 grupos de pacientes


k2 <- kmeans(df1, centers = 4, nstart = 25)

fviz_cluster(k2, geom = "point",  data = df1) + ggtitle("Número de grupos de estudiantes: 2")

fviz_cluster(k2, data = df1,
             palette=c("deeppink3", "magenta3"),
             ellipse.type = "euclid",
             star.plot = T,
             repel = T,
             ggtheme = theme())+ ggtitle("Número de grupos de pacientes: 2")

fviz_cluster(k2, df1, ellipse.type = "norm")

fviz_cluster(k2, df1, palette = "Set2", ggtheme = theme_minimal())

require(tibble)

k2 %>%
  extract_centroids()%>% as_tibble() %>% print(width=Inf)

kmeans_clusters <- 
  bind_cols(df1, cluster=k2$cluster)

kmeans_clusters %>%
  pivot_longer(-cluster) %>% 
  ggplot(aes(x = as.factor(cluster), y = value, fill = as.factor(cluster))) +
  geom_boxplot(show.legend = FALSE) +
  facet_wrap(vars(name), scales = "free") 

kmeans_clusters %>% 
  group_by(cluster) %>% 
  summarise(num_users = n()) %>% 
  mutate(pct_users = num_users / sum(num_users))

TC<-table(df$exam_score,k2$cluster)
TC

1-sum(diag(TC))/sum(TC)

###¿Y si lo hacemos con C.P.?

###Y si hacemos cluster, primero haciendo reduccion de dimension a traves de PCA
###3 componentes proporcionan alrededor del 89% de la varianza explicada y 4 un poco mas del 93%

kk<-eigen(cor(df1))
sum(kk$values[1:3])/sum(kk$values); sum(kk$values[1:4])/sum(kk$values)

###Clusters herarquicos


map <- c("No" = 0, "Yes" = 1)
map_diet <- c("Poor"=0,"Fair"=1,"Good"=2)
map_internet <- c("Poor"=0,"Average"=1,"Good"=2)


map_school <- c("None"=0,"High School"=1,"Bachelor"=2, "Master"=3)
map_gender <- c("Female"=1,"Male"=0,"Other"=2)


df$gender <- map_gender[df$gender]
df$diet_quality <- map_diet[df$diet_quality]
df$extracurricular_participation <- map[df$extracurricular_participation]
df$part_time_job <- map[df$part_time_job]
df$internet_quality <- map_internet[df$internet_quality]
df$parental_education_level <- map_school[df$parental_education_level]

df <- df %>% select(-exam_quantile)


df_pca_rec <- recipe(~ ., data = df) %>%
  update_role(exam_score, new_role = "id") %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), num_comp = 4)

df_pca_wf <- workflow() %>%
  add_recipe(df_pca_rec)

###Cluster herarquicos



df_pca_hier <- df_pca_wf %>%
  add_model(hier_clust(linkage_method = "ward.D")) %>%
  fit(data = df) %>%
  extract_fit_engine() %>%
  plot()

df_pca_hier <- df_pca_wf %>%
  add_model(hier_clust(linkage_method = "ward.D2")) %>%
  fit(data = df) %>%
  extract_fit_engine() %>%
  plot()

df_pca_hier <- df_pca_wf %>%
  add_model(hier_clust(linkage_method = "ward.D")) %>%
  fit(data = df) %>%
  extract_fit_engine() %>%
  fviz_dend(k = 4, main = "Dendograma basado en PCA: Liga Ward")%>%
  plot()

###K-Means

kmeans_specific <- k_means(num_clusters = 4) %>%
  set_engine("ClusterR")
  kmeans_specific

kmeans_workf <- workflow(df_pca_rec, kmeans_specific)

kmeans_proc <- fit(kmeans_workf, data = df)
kmeans_proc

kmeans_specific1 <- kmeans_specific %>% 
  set_args(num_clusters = tune())

kmeans_workf1 <- workflow(df_pca_rec, kmeans_specific1)
kmeans_workf1

set.seed(123)
boots <- bootstraps(df, times = 10)

tune_res <- tune_cluster(
  kmeans_workf1,
  resamples = boots
)

collect_metrics(tune_res)

extract_cluster_assignment(kmeans_proc) %>% print(n=Inf)

extract_centroids(kmeans_proc)

set.seed(123)
predict(kmeans_proc, new_data = slice_sample(as.data.frame(df),n = 10))

kmeans_summary <- kmeans_proc %>%
  extract_fit_summary()

kmeans_summary

tibble(
  orig_labels = kmeans_summary$orig_labels,
  standard_labels = kmeans_summary$cluster_assignments  ###Mismas asignaciones con distintas etiquetas
)

TC_CP<-table(df$exam_score,kmeans_summary$orig_labels)
TC_CP

1-sum(diag(TC_CP))/sum(TC_CP)

###Comparacion de clasificacion por K-Means: Todas las variables vs. CP

class_pca<-kmeans_summary$cluster_assignments

head(class_pca)

table(kmeans_clusters$cluster,class_pca)

kmeans_proc %>%
  silhouette_avg(df)

##Graficas finales

df <- df %>%
  mutate(
    exam_quantile = ntile(exam_score, 4),
    exam_quantile = factor(exam_quantile, 
                           labels = c("Q1 (Lowest)", "Q2(middle_1)", "Q3 (middle_2)","Q4 (highest)"))
  )

pca_cluster <-recipe(~ ., data = df) %>%
  update_role(exam_quantile, new_role = "id") %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), num_comp = 4) %>%
  prep(df) %>%
  bake(df)

pca_cluster

pca_clusters2 <- 
  bind_cols(pca_cluster, cluster=k2$cluster)

cluster_plot <- pca_clusters2 %>% 
  ggplot(mapping = aes(x = PC1, y = PC2)) +
  geom_point(aes(shape = factor(cluster)), size = 2) +
  scale_color_manual(values = c("darkorange","purple","cyan4","red"))

ggplotly(cluster_plot)

clust_spc_plot <- pca_clusters2 %>% 
  ggplot(mapping = aes(x = PC1, y = PC2)) +
  geom_point(aes(shape = factor(cluster), color = exam_quantile), size = 2, alpha = 0.8) +
  scale_color_manual(values = c("darkorange","purple","cyan4","red"))

clust_spc_plot

ggplotly(clust_spc_plot)

























