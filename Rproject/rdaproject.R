getwd()
setwd('C:\\Users\\jeeva\\Desktop\\rdaproject')

# Install
install.packages("tm")  # for text mining
install.packages("SnowballC") # for text stemming
install.packages("wordcloud") # word-cloud generator 
install.packages("RColorBrewer") # color palettes
install.packages("dplyr")
install.packages("ggpubr")
# Load
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")
library(tidyr)          # for data tidying
library(ggplot2)
library(lubridate)
library(dplyr)
library(ggpubr)
#1 How busy is it for Airbnb hosts in Toronto?
#unzipping the calendar file 
zz=gzfile('calendar.csv.gz','rt')   
cal=read.csv(zz,header=T)
head(cal)
dim(cal)
print(ncol(cal))
colnames(cal)
print(nrow(cal))
print(n_distinct(cal$date))
paste("We have",n_distinct(cal$date),"days and",n_distinct(cal$listing_id),"unique listings in the calendar data")
cal$date <- as.Date(cal$date,format = "%Y-%m-%d")
min(cal$date);max(cal$date)
paste("Calendar data covers one year time frame, that is, price and availability every day for the next one year.")
sum(is.na(cal))
summary(cal)
cal$busy=ifelse(cal$available=="t",0,1)
month = lubridate::month
calg=cal%>% group_by(monthgrouped=floor_date(date, "month")) %>%
  summarize(bus=mean(busy))
ggplot(calg, aes(monthgrouped,bus,label=calg$bus)) +
  geom_segment( aes(x=monthgrouped, xend=monthgrouped, y=0, yend=bus) , color=ifelse((calg$bus>=quantile(calg$bus,0.75)), "orange", "grey"), size=ifelse(calg$bus==max(calg$bus),.2, 0.2))  +
  geom_point(color=ifelse(calg$bus>=quantile(calg$bus,0.75), "orange", "grey")) +
  ylab("% busy")+
  xlab("Month")



#2 -----Price in months

cal$price <- as.numeric(gsub('\\$|,', '', cal$price))

# Reorder following the value of another column:

p <- ggplot(cal, aes(months.Date(cal$date), price)) +
  stat_summary(fun.y = "mean", geom = "point",na.rm=TRUE)
p
#3----weekdays
readr::locale("en")
Sys.setlocale("LC_TIME", "English")
cal$wd=(wday(cal$date,label=TRUE))
j=ggplot(cal, aes(x = wd, y =cal$price,na.rm=TRUE)) +
  stat_summary(fun.y="mean",geom="bar",na.rm=TRUE)+
  scale_y_continuous(breaks = scales::pretty_breaks(n = 20)) 
j
#LISTING-------
#4

listb=read.csv("listing.csv")
colnames(listb)
 #dplyr package
paste("There are total",n_distinct(listb$host_id) ,"unique host")
#neighborhood with highest airbnb host
nbhd= listb %>% group_by(neighbourhood_cleansed) %>%
  summarise(count=n_distinct(host_id))%>%
  arrange(desc(count))
k1=head(nbhd,20)
paste("The neighbourhood with highest number of airbnb host is",nbhd$neighbourhood_cleansed[1],"with",nbhd$count[1],"host")
k3=ggplot(k1, aes(x = neighbourhood_cleansed, y =count,na.rm=TRUE)) +
  stat_summary(fun.y="mean",geom="bar",na.rm=TRUE)+
  scale_y_continuous(breaks = scales::pretty_breaks(n = 20))+
  scale_fill_brewer(palette = "Set1")+
  ggpubr::rotate_x_text()+
  ggtitle("The top 20 Airbnb Neighbourhood")
k3
#price
listb$price <- as.numeric(gsub('\\$|,', '', listb$price))
summary(listb$price)
colnames(listb)
expensive=listb[listb$price==max(listb$price),]
paste("The most expensive Airbnb listing in Toronto is",expensive$name,"with one night price of",max(listb$price),"dollars",expensive$summary)
expensive$listing_url
expensive$picture_url
#5
#price vs neighbourhood
nbdp= listb %>% group_by(neighbourhood_cleansed) %>%
  summarise(cost=mean(price))%>%
  arrange(desc(cost))

gpl=ggplot(head(nbdp,20),aes(x=neighbourhood_cleansed,y=cost))+
             geom_bar(stat="identity")+
  scale_fill_brewer(palette = "Set2")+
  ggpubr::rotate_x_text()+
  ggtitle("The Expensive 20 Airbnb Neighbourhood")

gpl  
#AMENITIES
###--------------
listb$amenities=gsub('\\[|]|||"|"','',listb$amenities)
head(listb$amenities)
typeof(listb$amenities)
docs <- Corpus(VectorSource(listb$amenities))
inspect(docs)
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))

docs <- tm_map(docs, toSpace, "/")
docs <- tm_map(docs, toSpace, "@")
docs <- tm_map(docs, toSpace, "\\|")
# Convert the text to lower case
docs <- tm_map(docs, content_transformer(tolower))
# Remove numbers
docs <- tm_map(docs, removeNumbers)

# Remove english common stopwords
docs <- tm_map(docs, removeWords, stopwords("english"))
# Remove your own stop word
# specify your stopwords as a character vector
docs <- tm_map(docs, removeWords, c("blabla1", "blabla2")) 
# Remove punctuations
docs <- tm_map(docs, removePunctuation)
# Eliminate extra white spaces
docs <- tm_map(docs, stripWhitespace)
#---
dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)

v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
dreduced=head(d, 200)
#****
set.seed(1234)
wd=wordcloud(words = dreduced$word, freq = dreduced$freq, min.freq = 1,
             max.words=200, random.order=FALSE, rot.per=0.35, 
             colors=brewer.pal(8, "Dark2"))




#6
#room type
colnames(listb)

ppt=listb%>%  group_by(property_type) %>% 
  summarise(cost=mean(price))%>% 
  arrange(desc(cost))
#counts
n_distinct(listb$property_type)
ptc=listb%>%  group_by(property_type) %>% 
  summarise(counts=n_distinct(host_id))%>% 
  arrange(desc(counts))

#-------------------------------------------
{
library(scales)

label_data=ptc
label_data$counts=rescale(label_data$counts, to = c(1, 100))
label_data$id=seq(1:nrow(ptc))

# calculate the ANGLE of the labels
number_of_bar=nrow(label_data)

angle= 90 - 360 * (label_data$id-0.5) /number_of_bar     # I substract 0.5 because the letter must have the angle of the center of the bars. Not extreme right(1) or extreme left (0)


# calculate the alignment of labels: right or left
# If I am on the left part of the plot, my labels have currently an angle < -90
label_data$hjust<-ifelse( angle < -90, 1, 0)

# flip angle BY to make them readable
label_data$angle<-ifelse(angle < -90, angle+180, angle)
# ----- ------------------------------------------- ---- #


# Start the plot
m = ggplot(label_data, aes(x=property_type, y=counts)) +       # Note that id is a factor. If x is numeric, there is some space between the first bar
  
  # This add the bars with a blue color
  geom_bar(stat="identity", fill=alpha("skyblue", 0.7)) +
  
  # Limits of the plot = very important. The negative value controls the size of the inner circle, the positive one is useful to add size over each bar
  ylim(-50,120) +
  
  # Custom the theme: no axis title and no cartesian grid
  theme_minimal() +
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(rep(-1,4), "cm")      # Adjust the margin to make in sort labels are not truncated!
  ) +
  
  # This makes the coordinate polar instead of cartesian.
  coord_polar(start = 0) +
  
  # Add the labels, using the label_data dataframe that we have created before
  geom_text(data=label_data, aes(x=property_type, y=counts+10, label=property_type, hjust=hjust), color="black", fontface="bold",alpha=0.6, size=2.5, angle= label_data$angle, inherit.aes = FALSE ) 
m
}
#-------------------------------------
#outlier

outliers=boxplot(listb$price)$out
listo=listb[-which(listb$price %in% outliers),]
#-----
#roomtype vs price
#bed_type vs price
#7 & 8
ggplt= function(x,y=price)
{
p=ggplot(listo, aes(x,y)) + 
  geom_boxplot(
    
    # custom boxes
    color="blue",
    fill="blue",
    alpha=0.2,
    
 
    # custom outliers
    outlier.colour="red",
    outlier.fill="red",
    outlier.size=3
    
  )
p
}
ggplt(listo$room_type,listo$price)
ggplt(listo$bed_type,listo$price)
#REVIEWS
#9
rr=gzfile('reviews.csv.gz','rt')   
rvw=read.csv(rr,header=T)
colnames(rvw)
colnames(listb)
head(listb$review_scores_rating)
head(listb$review_scores_value)
#rating boxplot
outlier=boxplot(listb$review_scores_rating)$out
#forwdcld=sample_frac(listb,0.1)
worst=listb[which(listb$review_scores_rating %in% outlier),"host_id"]
best=na.omit(listb[listb$review_scores_rating == 100 ,"host_id"])
best
cmnt=rvw[which(rvw$listing_id %in% best),"comments"] 
cmnt
docs <- Corpus(VectorSource(cmnt))
inspect(docs)
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
docs <- tm_map(docs, toSpace, "/")
docs <- tm_map(docs, toSpace, "@")
docs <- tm_map(docs, toSpace, "\\|")
# Convert the text to lower case
docs <- tm_map(docs, content_transformer(tolower))
# Remove numbers
docs <- tm_map(docs, removeNumbers)

# Remove english common stopwords
docs <- tm_map(docs, removeWords, stopwords("english"))
# Remove your own stop word
# specify your stopwords as a character vector
docs <- tm_map(docs, removeWords, c("blabla1", "blabla2")) 
# Remove punctuations
docs <- tm_map(docs, removePunctuation)
# Eliminate extra white spaces
docs <- tm_map(docs, stripWhitespace)
#---
dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)

v <- sort(rowSums(m),decreasing=TRUE)
r <- data.frame(word = names(v),freq=v)
rreduced=head(r, 100)
#****
set.seed(1234)
wd=wordcloud(words = rreduced$word, freq = rreduced$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

