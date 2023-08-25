library(tidyverse)
library(rjson)
library(jsonlite)
library(data.table)

this.dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(this.dir)

set.seed(42)

# function for reading in .jsonl files
read_json_lines <- function(file){
  con <- file(file, open = "r")
  on.exit(close(con))
  jsonlite::stream_in(con, verbose = FALSE)
}

# select random rows from a dataframe
get_rand_rows = function(df, n_rows = 1) {
  df %>% slice(runif(n_rows, 0, length(df[,1])))
}

file_list = list.files("../../BBQ_full/data", full.names = T)
templates_list = list.files("../../BBQ_full/templates", full.names = T)

# just intersectional categories
#file_list = file_list[grepl("_x_", file_list)]

dat <- NULL
for(i in c(1:length(file_list))){
  temp = read_json_lines(file_list[i])
  additional_info = temp$additional_metadata
  ans_info = temp$answer_info
  ans_info = ans_info %>% rename("ans0_info" = ans0,
                                 "ans1_info" = ans1,
                                 "ans2_info" = ans2)
  temp2 = cbind(temp, additional_info)
  temp3 = cbind(temp2, ans_info)
  temp4 = temp3 %>% select(-additional_metadata, -answer_info)
  dat = rbind(dat,temp4)
}

templates = NULL
for(i in c(1:length(templates_list))){
  temp = read.csv(templates_list[i],stringsAsFactors = F)
  if(!grepl("vocabulary|Filler",templates_list[i])){
    temp2 <- temp %>%
      select(needs_val,Q_id,Category)
    templates = rbind(templates,temp2)
  }
}
templates2 = templates %>% 
  rename("category" = "Category", "question_index" = "Q_id")%>%
  mutate(category=ifelse(category=="DisabilityStatus","Disability_status",
                         ifelse(category=="GenderIdentity","Gender_identity",
                                ifelse(category=="PhysicalAppearance", "Physical_appearance",
                                       ifelse(category=="RaceEthnicity", "Race_ethnicity",
                                              ifelse(category=="SexualOrientation","Sexual_orientation",
                                                     ifelse(category=="Sexual orientation","Sexual_orientation",
                                                            ifelse(category=="Religion ","Religion",category))))))))%>%
  filter(category!="")%>%
  group_by(category,question_index,needs_val)%>%
  summarise(count=n())%>%
  select(-count)%>%
  filter(grepl("_x_", category)) # for intersectional only

fillers = read.csv("../../BBQ_full/templates/new_templates - Filler_items.csv", stringsAsFactors = F)
fillers = fillers %>% 
  filter(!is.na(Q_id))#%>%
  #filter(Q_id >= 200) # for just intersectional categories

all_cats = unique(dat$category)

ex_dat = NULL
for(i in c(1:length(all_cats))){
  temp <- dat %>%
    filter(category == all_cats[i])
  
  Q_ids = unique(temp$question_index)
  for(j in c(1:length(Q_ids))){
    temp2 <- temp %>%
      filter(question_index == Q_ids[j])
    
    neg_ambig = sample_n(temp2 %>% filter(question_polarity=="neg" & context_condition == "ambig"),1)
    neg_disambig = sample_n(temp2 %>% filter(question_polarity=="neg" & context_condition == "disambig"),1)
    nonneg_ambig = sample_n(temp2 %>% filter(question_polarity=="nonneg" & context_condition == "ambig"),1)
    nonneg_disambig = sample_n(temp2 %>% filter(question_polarity=="nonneg" & context_condition == "disambig"),1)
    ex_dat = do.call("rbind", list(ex_dat, neg_ambig, neg_disambig, nonneg_ambig, nonneg_disambig))
  }
}

ex_dat2 <- merge(templates2,ex_dat,all=T)

ex_dat3 <- ex_dat2 %>% 
  filter(!is.na(example_id))%>%
  #filter(needs_val == "Y")%>%
  #select(-needs_val)
  sample_n(300)
  

# ex_dat = NULL
# for(i in c(1:length(all_cats))){
#   temp <- dat %>%
#     filter(category == all_cats[i])
#   Q_ids = unique(temp$question_index)
#   for(j in c(1:length(Q_ids))){
#     temp2 <- temp %>%
#       filter(question_index == Q_ids[j])#%>%
#       # mutate(ans0_info = sapply(ans0_info, "[", 2),
#       #        ans1_info = sapply(ans1_info, "[", 2),
#       #        ans2_info = sapply(ans2_info, "[", 2))
#     # if(all_cats[i] %in% c("Nationality", "Race_ethnicity", "Religion")){
#     #   temp2 = temp2 %>% 
#     #     filter(tolower(stereotyped_groups)==tolower(ans0_info)|
#     #             tolower(stereotyped_groups)==tolower(ans1_info)|
#     #             tolower(stereotyped_groups)==tolower(ans1_info))
#     # }
#     # if(nrow(temp2)<4){
#     #   temp2 <- temp %>%
#     #     filter(question_index == Q_ids[j])%>%
#     #     mutate(ans0_info = sapply(ans0_info, "[", 2),
#     #            ans1_info = sapply(ans1_info, "[", 2),
#     #            ans2_info = sapply(ans2_info, "[", 2))%>%
#     #     filter(grepl(stereotyped_groups[[1]], ans0) |
#     #            grepl(stereotyped_groups[[1]], ans1) |
#     #            grepl(stereotyped_groups[[1]], ans2))
#     # }
#     # if(nrow(temp2)<4){
#     #   #print(paste0("still error with ",all_cats[i]," template ",Q_ids[j]))
#     #   temp2 <- temp %>%
#     #     filter(question_index == Q_ids[j])%>%
#     #     mutate(ans0_info = sapply(ans0_info, "[", 2),
#     #            ans1_info = sapply(ans1_info, "[", 2),
#     #            ans2_info = sapply(ans2_info, "[", 2))
#     # } else {
#       neg_ambig = get_rand_rows(temp2 %>% filter(question_polarity=="neg" & context_condition == "ambig"))
#       neg_disambig = get_rand_rows(temp2 %>% filter(question_polarity=="neg" & context_condition == "disambig"))
#       nonneg_ambig = get_rand_rows(temp2 %>% filter(question_polarity=="nonneg" & context_condition == "ambig"))
#       nonneg_disambig = get_rand_rows(temp2 %>% filter(question_polarity=="nonneg" & context_condition == "disambig"))
#       ex_dat = do.call("rbind", list(ex_dat, neg_ambig, neg_disambig, nonneg_ambig, nonneg_disambig))
#     #}
#   }
# }

colnames(fillers)<-tolower(colnames(fillers))
fills2 <- fillers %>%
  rename("question_index" = "q_id")%>%
  gather(question_polarity, question, -question_index, - category, - context, - ans0, -ans1, -ans2, -label_neg_ans, -label_non_neg_ans, -notes) %>%
  mutate(question_polarity = case_when(question_polarity=="question_negative_stereotype" ~ "neg",
                                       question_polarity=="question_non_negative" ~ "nonneg"))%>%
  mutate(label = case_when(question_polarity=="neg" ~ label_neg_ans,
                           question_polarity=="nonneg" ~ label_non_neg_ans))%>%
  rename("context_condition" = notes)%>%
  select(-label_neg_ans,-label_non_neg_ans)

fills2$subcategory = NA
fills2$example_id = NA
fills2$stereotyped_groups = NA
fills2$version = NA
fills2$source = NA
fills2$ans0_info = NA
fills2$ans1_info = NA
fills2$ans2_info = NA

all_examples = rbind(sample_n(fills2,50),ex_dat3%>%select(-needs_val))
all_examples <- apply(all_examples,2,as.character)

# save to make a few manual edits
write.csv(all_examples, "all_examples_finalval.csv", row.names = F)  

examples = read.csv("all_examples_finalval.csv", stringsAsFactors = F)

# give unique identifier to each list
examples$u_id<-seq.int(nrow(examples))

# save uids
write.csv(examples, "all_examples_uid_finalval.csv", row.names = F) 

# initialize data frame needed in the loop
already_used = c(0)
FullTurkDataset = NULL

# randomize order
examples2 <- sample_n(examples, nrow(examples))

# same loop as above, but don't latin square
for(i in c(1:ceiling(nrow(examples2)/5))){
  # make sure used items are taken out
  newData = examples2%>%
    filter(!u_id %in% already_used)
  # get nonduplicates of categories
  tempData = newData[!duplicated(newData[c("category")]),]
  print(paste0("tempData run",i,": ",nrow(tempData)))
  # just in case the nonduplicates is less than 4, can pull from the full dataset
  if(nrow(tempData)>=5){
    subsetTempData = sample_n(tempData, 5)
  } else {
    print(paste0("using newData, run",i,": ",nrow(newData)))
    subsetTempData = newData[1:5,] 
  }
  # add the selected 5 rows to already_used so they can be taken out on the next run of the loop
  already_used = c(already_used,unique(subsetTempData$u_id))
  # select only the columns needed
  tempData2 <- subsetTempData %>%
    select(context,question,u_id,ans0,ans1,ans2,label,category,example_id)
  turkDataFile_temp = data.frame("dummy" = c(0))
  for(j in c(1:nrow(tempData2))){
    tempData3 <- tempData2[j,]
    colnames(tempData3) <- c(paste0('q',j,"_input"), paste0('q',j,"_question"),paste0('q',j,"_uid"),
                             paste0('q',j,"_ans_0"),paste0('q',j,"_ans_1"),paste0('q',j,"_ans_2"),
                             paste0('q',j,"_label"),paste0('q',j,"_cat"),paste0('q',j,"_ex_id"))
    turkDataFile_temp = cbind(turkDataFile_temp,tempData3)
  }
  # add rows to the full dataframe for mturk upload
  FullTurkDataset=rbind(FullTurkDataset,turkDataFile_temp)
}

fulldata <- FullTurkDataset %>% select(-dummy)

write.csv(fulldata,"../for_upload/donotupload_full_data_for_mturk_full_finalval.csv",row.names = F)
write.csv(fulldata[1:25,],"../for_upload/full_data_for_mturk_1_finalval.csv",row.names = F)
write.csv(fulldata[26:50,],"../for_upload/full_data_for_mturk_2_finalval.csv",row.names = F)
write.csv(fulldata[51:70,],"../for_upload/full_data_for_mturk_3_finalval.csv",row.names = F)
#write.csv(fulldata[101:127,],"../for_upload/full_data_for_mturk_4_round2.csv",row.names = F)
