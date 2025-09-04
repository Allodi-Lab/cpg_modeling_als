R Code
#VGLUT2
setwd("/users/.../...")
mice<-read.table ("VGlut2.csv", header=T, sep="," )
library(ggplot2)
 
#Graphs
plot2 <-ggplot(data= mice, aes(x=Treatment, y=Avg.Mean)) +
  geom_bar(stat = "summary",fun=mean, aes(color = Treatment,fill=Treatment), alpha = 0.01, size=1) +
  geom_errorbar(stat="summary", fun.data=mean_se, width = 0.4, position = "dodge")+
  theme(panel.background = element_rect(fill = "white"), axis.line = element_line(color="black"))+
  theme(text = element_text(family = "Arial"))+
#  scale_color_manual(values=color_palette)+
#  scale_fill_manual(values=color_palette)+
  geom_jitter(position = position_jitter(width = .2),alpha=0.3)+
  theme(axis.title.x=element_blank(),  text = element_text(size=15),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
 
ggp2<-plot2+labs(x="Treatment",y="Average Area")
 
##Chx10
setwd("/users/.../...")
mice<-read.table ("Chx10.csv", header=T, sep="," )
 
#Graphs
plot <-ggplot(data= mice, aes(x=Treatment, y=Avg.Count)) +
  geom_bar(stat = "summary",fun=mean, aes(color = Treatment,fill=Treatment), alpha = 0.01, size=1) +
  geom_errorbar(stat="summary", fun.data=mean_se, width = 0.4, position = "dodge")+
  theme(panel.background = element_rect(fill = "white"), axis.line = element_line(color="black"))+
  theme(text = element_text(family = "Arial"))+
  geom_jitter(position = position_jitter(width = .2),alpha=0.3)+
  theme(axis.title.x=element_blank(),  text = element_text(size=15),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
 
ggp<-plot+labs(x="Treatment",y="V2a Neuron Count")
 
#T Test
result = t.test(Avg.Count~Treatment, data = mice, paired=FALSE)
result