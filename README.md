 ## Алгоритм 1NN

Алгоритм ближайшего соседа - 1NN  относит классифицируемый объект ![screenshot of sample](https://github.com/LenuraA/ML1/blob/master/формула1.png) к тому классу , к которому относится его ближайший сосед.
 ![screenshot of sample]( https://github.com/LenuraA/ML1/blob/master/формула2.png) 

## Приемущество: 
 Простота 

## Недостатки: 
  Неустойчивость к погрешностям — выбросам.
 Отсутствие параметров, которые можно было бы настраивать по выборке. Алгоритм полностью зависит от того, насколько удачно выбрана
метрика *ρ*.



 1. Нужно задать метрическую функцию. 
 2. Обучающая выборка сортируется в порядке увеличения расстояния от классифицируемого элемента.  
 3. Классифицируемый элемент относим к классу, к которому принадлежит ближайший элемент(первый в отсортированной выборке). 
  
  
 
  Метрические алгоритмы классификации с обучающей выборкой *Xl* относят объект *u* к тому классу *y*, для которого суммарный вес ближайших обучающих объектов  ![screenshot of sample](https://github.com/LenuraA/ML1/blob/master/формула3.png) максимален: 
  ![screenshot of sample](https://github.com/LenuraA/ML1/blob/master/формула4.png)
  
  
   , где весовая функция *w(i, u)* оценивает степень важности *i*-го соседа для классификации объекта *u*.
   Функция ![screenshot of sample]( https://github.com/LenuraA/ML1/blob/master/формула3.png) называется оценкой близости объекта *u* к классу *y*. Выбирая различную весовую функцию *w(i, u)* можно получать различные метрические классификаторы.
  
  ```R
  distances = function(xl, data, k) { # возвращает отсортированный набор данных по метрике для объекта 
 cases = dim(data)[1]
 features = dim(data)[2]-1
 dists = matrix(0, cases, 2) # создаем матрицу расстояний 
   for (i in 1:cases) {
     cost = k(xl, data[i,1:features])
     dists[i,] = c(cost, i)
   }
   idx = order(dists[,1])
   data[dists[idx,2],]
 }
NN = function(xl, data, k=dist) 
 { # находит 1-ближайшего соседа 
 sorted = distances(xl, data, k)
   sorted[1,features+1]
 }
 ```
 
 ![screenshot of sample](https://github.com/LenuraA/ML1/blob/master/1.1nn.png)


## Алгоритм k ближайших соседей 
 Алгоритм k ближайших соседей – kNN относит объект *u* к тому классу,
элементов которого больше среди *k* ближайших соседей  ![screenshot of sample](https://github.com/LenuraA/ML1/blob/master/формула6.png)


 Алгоритм работает следующим образом: пусть дан классифицируемый объект *u* и обучающая выборка . Требуется определить класс объекта *u* на основе данных из обучающей выборки. Для этого:

1. Вся выборка  сортируется по возрастанию расстояния от объекта *u* до каждого объекта выборки.
2. Проверяются классы *k* ближайших соседей объекта *u*. Класс, встречаемый наиболее часто, присваивается объекту *u*. Для оценки близости классифицируемого объекта *u* к классу  алгоритм kNN использует следующую функцию: ![screenshot of sample](https://github.com/LenuraA/ML1/blob/master/формула7.png) , где *i* - порядок соседа по расстоянию к классифицируемому объекту *u*.

```R
kNN = function(xl, data, k, m=dist) { # находит k-ближайших соседей  
 sorted = distances(xl, data, m)
   
   n = 10 
   counts = rep(0, times=n)
 features = dim(data)[2]-1
 for (i in 1:k) {
     cls = sorted[i,features+1]
     counts[cls] = counts[cls] + 1
   }
   argmax = 1
   for (i in n) {
     if (counts[argmax] < counts[i]) {
       argmax = i
     }
   }
   
   cls[argmax]
 }
 ```
 
 ![screenshot of sample](https://github.com/LenuraA/ML1/blob/master/KNN.png)

Алгоритм kNN выглядит  качествеено. Для того чтобы привести более точное обоснование чем kNN лучше в этом случае, чем 1NN, следует прибегнуть к скользящему контролю.

Для поиска оптимальных параметров для каждого из рассматриваемых ниже метрических алгоритмов используется LOO -- leave-one-out (критерий скользящего контроля), который состоит в следующем:

1.Исключать объекты *x(i)* из выборки *Xl* по одному, получится новая выборка без объекта *x(i)*.

2.Запускать алгоритм от объекта *u*, который нужно классифицировать, на выборке *Xl_1*.

3.Завести переменную *Q* (накопитель ошибки, изначально Q = 0) и, когда алгоритм ошибается, Q = Q + 1.

4.Когда все объекты *x(i)* будут перебраны, вычислить LOO = Q / l (l -- количество объектов выборки).

При минимальном значении LOO получим оптимальный параметр алгоритма.

```R
LOO = function(xl,class) {
  n = dim(xl)[1];
  loo = rep(0, n-1) 
  
    for(i in 1:n){
      X=xl[-i, 1:3]
      u=xl[i, 1:2]
      orderedXl <- sortObjectsByDist(X, u)
      
      for(k in 1:(n-1)){
        
        test=knn(X,u,k,orderedXl)
        if(colors[test] != colors[class[i]]){
          loo[k] = loo[k]+1;
        
      }
    } 
    }
  
  loo = loo / n
  x = 1:length(loo)
  y = loo
  plot(x, y,main ="LOO for KNN(k)", xlab="k", ylab="LOO", type = "l")

  min = which.min(loo)
  lOOmin=round(loo[min],3)
  points(min, loo[min], pch = 21, col = "red",bg = "red")
  label = paste("   K = ", min, "\n", "   LOO = ", lOOmin, sep = "")
  xmin = 3*min;
  text(xmin, lOOmin, labels = label, pos = 3, col = "red")
  map(min);
}
```

![screenshot of sample](https://github.com/LenuraA/ML1/blob/master/LOO.png)
