import streamlit as st
from streamlit_option_menu import option_menu
import pages as pg
import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Set page title
st.set_page_config(page_title="Blue Horizontal Menu", layout="wide")
st.markdown("<h4 style= 'text-align: center; background-color:#f08c44; color:#fff;border-radius: 5px;' > Status = InProgress</h4>",unsafe_allow_html=True )
st.markdown(
    """
    <style>
    .menu {
        border-radius: 5px;
        display: flex;
        list-style: none;
       padding: 20px;
       margin: 10px;
    }
    .active {
        background-color: red !important;
    }
    .option_menu {
    background: #007ACC;
    color: #fff;
    outline: none;
    padding: 0.75em 0.75em 0.75em 2.25em;
    position: relative;
    text-decoration: none;
    transition: background 0.2s linear
    }
    .menu-item a {
        color: white;
        text-decoration: none;
    }
    .menu-item a:hover {
        text-decoration: underline;
        background: #6c8fc4;
    }
    .menu-item:focus:after,
   .menu-item:focus,
   .menu-item.is-active:focus {
    background: #fff;
    color: #fff;
     }
        .menu-item:after,
        .menu-item:before {
        background: #007ACC;
        bottom: 0;
        clip-path: polygon(50% 50%, -50% -50%, 0 100%);
        content: "";
        left: 100%;
        position: absolute;
        top: 0;
        transition: background 0.2s linear;
        width: 2em;
        z-index: 1;
        }
        
        .menu-item:last-child {
            border-right: none;
        }
        .active {
        background-color: #fff;
           
        }
          .menu-item:before {
    background: #fff;
    margin-left: 1px;
  }
  [data-testid=stSidebar] {
        background-color: #007ACC;
         color:#fff;
           }
.center {
    text-align: center;

}
     
    </style>
    """,
    unsafe_allow_html=True
)



pages = ["info","Similarcases","Indicators","Network","Feedback"]

selected_menu = option_menu(
    menu_title=None,
    options=["info","Similarcases","Indicators","Network","Feedback"],
    icons=["info-lg","transparency","app-indicator","people","pencil"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
     "container" : {"padding": "0!importatnt",  "border-radius": "5px", "background-color" :"#007ACC" , "display": "flex"},
     "icon":{"color": "white", "font-size": "25px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee", "color": "white"},
        "nav-link-selected": {"background-color": "orange"},
     }

)


with st.sidebar:
    st.write("Search")
    st.form("search")
    keyword=st.text_input("")

if selected_menu:
    if selected_menu == "info":
        import streamlit as st
        import pandas as pd

        st.title("Claim Info")

        # Sample data
        data = {
            "Process": ["Commercial_motor_underwriting"],
            "Product": ["Commercial motor/Auto"],
            "Label": ['Testinsurer'],
            "Branch": ["US"],
            "Client ID": [12345674],
            "Claim ID": [12345674],
            "Policy ID": [12345674],
            "Underwriting ID": [12345674],
            "Underwriting Type": ["Policy"],
            "Status": ["New"],
            "Impact": ["New"],
            "Investigator": ["XXXX"],
            "Date": ["02/03/2014"],

        }
        vertical_table = pd.DataFrame(data).T
        styled_table = vertical_table.style.set_properties(**{
    'font-size': '14px',
    'text-align': 'center',
    'width':'500px',
    'color':'#ff6600',
    'background-color':'#e6faff'
})

       # Convert styled DataFrame to HTML
        styled_html = styled_table.to_html(escape=False)

        # Display the styled table using Markdown
        st.markdown(styled_html, unsafe_allow_html=True)


    elif selected_menu == "Similarcases":
        

        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns


        from sklearn.metrics import  confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, accuracy_score, classification_report, confusion_matrix, auc
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.preprocessing import StandardScaler, PowerTransformer

        import streamlit as st

        import warnings
        warnings.filterwarnings("ignore")



        # Title
        st.markdown("<h3 style= 'text-align: left;' > Comparing Case #1234 with previous cases</h3>",unsafe_allow_html=True)
        
        import streamlit as st
        import altair as alt
        import pandas as pd

        # Define the data for the grouped bar chart
        data = pd.DataFrame({
            'Factor': ['Tax', 'Income', 'Picture', 'Time', 'Friend'],
            'Case#1234': [89, 45, 23, 67, 54],
            'Case #4567': [32, 45, 54, 65, 12],
            'Case #7654': [23, 44, 54, 67, 50]
        })

        # Melt the data for Altair
        melted_data = data.melt('Factor', var_name='Group', value_name='Percentage')

        # Create the grouped bar chart
        chart = alt.Chart(melted_data).mark_bar().encode(
            x=alt.X('Factor:N', title='Factor', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Percentage:Q', title='Percentage'),
            color=alt.Color('Group:N', scale=alt.Scale(range=['#1f77b4', '#ff7f0e', '#2ca02c']))
        ).properties(
            width=alt.Step(100),
            height=300
        ).configure_axis(
            grid=False
        )

        # Display the chart
        st.altair_chart(chart)





        # Read Data into a Dataframe
        df = pd.read_csv('creditcard.csv')

        st.set_option('deprecation.showPyplotGlobalUse', False)
        # --- 1 CHECKBOX ---
        # Print description of the initial data and shape
        
        



        # --- 2 CHECKBOX ---
        if st.sidebar.checkbox('Show the analysis'):
            
            fraud = df[df.Class == 1]
            valid = df[df.Class == 0]

            outlier_percentage=(df.Class.value_counts()[1]/df.Class.value_counts()[0])*100

            st.header('Univariate analysis')

            st.write('Fraud Cases: ', len(fraud))
            st.write('Valid Cases: ', len(valid))
            st.write('Compare the values for both transactions: \n', df.groupby('Class').mean())
            st.write('Fraudulent transactions are: %.3f%%'%outlier_percentage)


            # Method to compute countplot of given dataframe parameters:
            # - data(pd.Dataframe): Input Dataframe
            # - feature(str): Feature in Dataframe
            def countplot_data(data, feature):
                plt.figure(figsize=(10,10))
                sns.countplot(x=feature, data=data)
                plt.show()

            # Method to construct pairplot of the given feature wrt data parameters:
            # - data(pd.DataFrame): Input Dataframe
            # - feature1(str): First Feature for Pair Plot
            # - feature2(str): Second Feature for Pair Plot
            # - target: Target or Label (y)
            def pairplot_data_grid(data, feature1, feature2, target):
                sns.FacetGrid(data, hue=target).map(plt.scatter, feature1, feature2).add_legend()
                plt.show()

            st.subheader('Transaction ratio:')
            st.pyplot(countplot_data(df, df.Class))

            st.subheader('The relationship of fraudulent transactions with the amount of money:\n')
            st.pyplot(pairplot_data_grid(df, "Time", "Amount", "Class"))
            


            st.header('Bivariate Analysis')
            
            st.write('Fraud: ', df.Time[df.Class == 1].describe())
            st.write('Not fraud: ', df.Time[df.Class == 0].describe())

            
            def graph1():
                f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
                bins = 50

                ax1.hist(df.Time[df.Class == 1], bins = bins)
                ax1.set_title('Fraud')

                ax2.hist(df.Time[df.Class == 0], bins = bins)
                ax2.set_title('Not Fraud')

                plt.xlabel('Time (Sec.)')
                plt.ylabel('Number of Transactions')
                plt.show()


            def graph2():
                f, axes = plt.subplots(ncols=2, figsize=(16,10))
                colors = ['#C35617', '#FFDEAD']

                sns.boxplot(x="Class", y="Amount", data=df, palette = colors, ax=axes[0], showfliers=True)
                axes[0].set_title('Class vs Amount')

                sns.boxplot(x="Class", y="Amount", data=df, palette = colors, ax=axes[1], showfliers=False)
                axes[1].set_title('Class vs Amount without outliers')

                plt.show()

            
            def graph3():
                fig, ax = plt.subplots(1, 2, figsize=(18,4))

                amount_val = df['Amount'].values
                time_val = df['Time'].values

                sns.distplot(amount_val, ax=ax[0], color='b')
                ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
                ax[0].set_xlim([min(amount_val), max(amount_val)])

                sns.distplot(time_val, ax=ax[1], color='r')
                ax[1].set_title('Distribution of Transaction Time', fontsize=14)
                ax[1].set_xlim([min(time_val), max(time_val)])

                plt.show()


            st.pyplot(graph1())
            st.pyplot(graph2())
            st.pyplot(graph3())



            st.header('Multivariate Analysis')


            # Plot relation with different scale
            def graph4(): 
                df1 = df[df['Class']==1]
                df2 = df[df['Class']==0]
                fig, ax = plt.subplots(1,2, figsize=(15, 5))

                ax[0].scatter(df1['Time'], df1['Amount'], color='red', marker= '*', label='Fraudrent')
                ax[0].set_title('Time vs Amount')
                ax[0].legend(bbox_to_anchor =(0.25, 1.15))

                ax[1].scatter(df2['Time'], df2['Amount'], color='green', marker= '.', label='Non Fraudrent')
                ax[1].set_title('Time vs Amount')
                ax[1].legend(bbox_to_anchor =(0.3, 1.15))

                plt.show()


            def graph5():
                sns.lmplot(x='Time', y='Amount', hue='Class', markers=['x', 'o'], data=df, height=6)
            

            # plot relation in same scale
            def graph6():
                g = sns.FacetGrid(df, col="Class", height=6)
                g.map(sns.scatterplot, "Time", "Amount", alpha=.7)
                g.add_legend()
            

            st.pyplot(graph4())
            st.pyplot(graph5())
            st.pyplot(graph6())  
        # --- 2 CHECKBOX ---



        # --- 4 CHECKBOX ---
        if st.sidebar.checkbox('Compare algorithms'):
            # --- TRAIN AND TEST SPLIT ---
            # Putting feature variables into X
            X = df.drop(['Class'], axis=1)

            # Putting target variable to y
            y = df['Class']


            # Splitting data into train and test set 80:20
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 42)
            # --- TRAIN AND TEST SPLIT ---


            # --- FEATURE SCALING ---
            # Instantiate the Scaler
            scaler = StandardScaler()


            # Fit the data into scaler and transform
            X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])


            # Transform the test set
            X_test['Amount'] = scaler.transform(X_test[['Amount']])


            # Checking the Skewness
            # Listing the columns
            cols = X_train.columns
            # --- FEATURE SCALING ---


            # --- Mitigate skwenes with PowerTransformer ---
            # Instantiate the powertransformer
            pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=False)

            # Fit and transform the PT on training data
            X_train[cols] = pt.fit_transform(X_train)

            # Transform the test set
            X_test[cols] = pt.transform(X_test)
            # --- Mitigate skwenes with PowerTransformer ---


            def visualize_confusion_matrix(y_test, y_pred):
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(7, 5))
                sns.heatmap(cm, annot=True, fmt='g', cmap='Oranges',
                            xticklabels=['No Credit Card Fraud Dection','Credit Card Fraud Dection'], 
                            yticklabels=['No Credit Card Fraud Dection','Credit Card Fraud Dection'])
                plt.title('Accuracy: {0:.4f}'.format(accuracy_score(y_test, y_pred)))
                plt.ylabel('True Values')
                plt.xlabel('Predicted Values')
                plt.show()
                
                st.write("\n")
                st.write("Classification Report:")
                st.text(classification_report(y_test, y_pred))
                return

            
            def ROC_AUC(Y, Y_prob):
                # caculate roc curves
                fpr, tpr, threshold = roc_curve(Y, Y_prob)
                # caculate scores
                model_auc = roc_auc_score(Y, Y_prob)
                # plot roc curve for the model
                plt.figure(figsize=(16, 9))
                plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
                plt.plot(fpr, tpr, marker='.', label='Model - AUC=%.3f' % (model_auc))
                # show axis labels and the legend
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend()
                plt.show(block=False)
                return


            # --- START Logistic regression ---
            st.header('Logistic Regression')
            # --- START Training the Logistic Regression Model on the Training set ---
            st.subheader('Training the Logistic Regression Model on the Training set')
            

            LR_model = LogisticRegression(random_state = 0)
            LR_model.fit(X_train, y_train)
            y_train_pred = LR_model.predict(X_train)
            y_test_pred = LR_model.predict(X_test)
            acc1 = accuracy_score(y_test, y_test_pred)


            # Train Score
            st.write('Recall score: %0.4f'% recall_score(y_train, y_train_pred))
            st.write('Precision score: %0.4f'% precision_score(y_train, y_train_pred))
            st.write('F1-Score: %0.4f'% f1_score(y_train, y_train_pred))
            st.write('Accuracy score: %0.4f'% accuracy_score(y_train, y_train_pred))
            st.write('AUC: %0.4f' % roc_auc_score(y_train, y_train_pred))

            # Train Predictions
            st.pyplot(visualize_confusion_matrix(y_train, y_train_pred))


            st.pyplot(ROC_AUC(y_train, y_train_pred))
            # --- END Training the Logistic Regression Model on the Training set ---


            # --- START Training the Logistic Regression Model on the Testing set ---
            st.subheader('Training the Logistic Regression Model on the Testing set')


            # Test score
            st.write('Recall score: %0.4f'% recall_score(y_test, y_test_pred))
            st.write('Precision score: %0.4f'% precision_score(y_test, y_test_pred))
            st.write('F1-Score: %0.4f'% f1_score(y_test, y_test_pred))
            st.write('Accuracy score: %0.4f'% accuracy_score(y_test, y_test_pred))
            st.write('AUC: %0.4f' % roc_auc_score(y_test, y_test_pred))


            # Test Predictions
            st.pyplot(visualize_confusion_matrix(y_test, y_test_pred))


            st.pyplot(ROC_AUC(y_test, y_test_pred))
            # --- END Training the Logistic Regression Model on the Testing set ---


            # Result
            st.header('Results')
            st.subheader('Training set')
            st.text('- Recall score: 0.6397\n- Precision score: 0.8688\n- F1-Score: 0.7368\n- Accuracy score: 0.9992\n- AUC: 0.8198')
            

            st.subheader('Testing set')
            st.text('- Recall score: 0.5556\n- Precision score: 0.9091\n- F1-Score: 0.6897\n- Accuracy score: 0.9992\n- AUC: 0.7777')
            # --- END Logistic regression ---



            # --- START Naive Bayes ---
            st.header('Naive Bayes')

            
            # --- START Training the Naive Bayes Model on the Training set ---
            st.subheader('Training the Naive Bayes Model on the Training set')


            NB_model = GaussianNB()
            NB_model.fit(X_train, y_train)
            y_train_pred = NB_model.predict(X_train)
            y_test_pred = NB_model.predict(X_test)
            acc2 = accuracy_score(y_test, y_test_pred)


            # Train Score
            st.write('Recall score: %0.4f'% recall_score(y_train, y_train_pred))
            st.write('Precision score: %0.4f'% precision_score(y_train, y_train_pred))
            st.write('F1-Score: %0.4f'% f1_score(y_train, y_train_pred))
            st.write('Accuracy score: %0.4f'% accuracy_score(y_train, y_train_pred))
            st.write('AUC: %0.4f' % roc_auc_score(y_train, y_train_pred))


            # Train Predictions
            st.pyplot(visualize_confusion_matrix(y_train, y_train_pred))


            st.pyplot(ROC_AUC(y_train, y_train_pred))
            # --- END Training the Naive Bayes Model on the Training set ---


            # --- START Training the Naive Bayes Model on the Testing set ---
            st.subheader('Training the Naive Bayes Model on the Testing set')


            # Test score
            st.write('Recall score: %0.4f'% recall_score(y_test, y_test_pred))
            st.write('Precision score: %0.4f'% precision_score(y_test, y_test_pred))
            st.write('F1-Score: %0.4f'% f1_score(y_test, y_test_pred))
            st.write('Accuracy score: %0.4f'% accuracy_score(y_test, y_test_pred))
            st.write('AUC: %0.4f' % roc_auc_score(y_test, y_test_pred))


            # Test Predictions
            st.pyplot(visualize_confusion_matrix(y_test, y_test_pred))


            st.pyplot(ROC_AUC(y_test, y_test_pred))
            # --- END Training the Naive Bayes Model on the Testing set ---


            # Result
            st.header('Results')
            st.subheader('Training set')
            st.text('- Recall score: 0.8277\n- Precision score: 0.0604\n- F1-Score: 0.1125\n- Accuracy score: 0.9780\n- AUC: 0.9030')
            

            st.subheader('Testing set')
            st.text('- Recall score: 0.7778\n- Precision score: 0.0523\n- F1-Score: 0.0980\n- Accuracy score: 0.9773\n- AUC: 0.8777')
            # --- END Naive Bayes ---

            
            
            # --- START Decision tree ---
            st.header('Decision tree')

            
            # --- START Training the Decision tree Model on the Training set ---
            st.subheader('Training the Decision tree Model on the Training set')


            DTR_model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
            DTR_model.fit(X_train, y_train)
            y_train_pred = DTR_model.predict(X_train)
            y_test_pred = DTR_model.predict(X_test)
            acc4 = accuracy_score(y_test, y_test_pred)


            # Train Score
            st.write('Recall score: %0.4f'% recall_score(y_train, y_train_pred))
            st.write('Precision score: %0.4f'% precision_score(y_train, y_train_pred))
            st.write('F1-Score: %0.4f'% f1_score(y_train, y_train_pred))
            st.write('Accuracy score: %0.4f'% accuracy_score(y_train, y_train_pred))
            st.write('AUC: %0.4f' % roc_auc_score(y_train, y_train_pred))


            st.pyplot(visualize_confusion_matrix(y_train, y_train_pred))

            st.pyplot(ROC_AUC(y_train, y_train_pred))
            # --- END Training the Decision tree Model on the Training set ---


            # --- START Training the Decision tree Model on the Testing set ---
            st.subheader('Training the Decision tree Model on the Testing set')


            st.write('Recall score: %0.4f'% recall_score(y_test, y_test_pred))
            st.write('Precision score: %0.4f'% precision_score(y_test, y_test_pred))
            st.write('F1-Score: %0.4f'% f1_score(y_test, y_test_pred))
            st.write('Accuracy score: %0.4f'% accuracy_score(y_test, y_test_pred))
            st.write('AUC: %0.4f' % roc_auc_score(y_test, y_test_pred))


            st.pyplot(visualize_confusion_matrix(y_test, y_test_pred))

            st.pyplot(ROC_AUC(y_test, y_test_pred))
            # --- END Training the Decision tree Model on the Testing set ---


            # Result
            st.header('Results')
            st.subheader('Training set')
            st.text('- Recall score: 1.0000\n- Precision score: 1.0000\n- F1-Score: 1.0000\n- Accuracy score: 1.0000\n- AUC: 1.0000')
            

            st.subheader('Testing set')
            st.text('- Recall score: 0.6889\n- Precision score: 0.7561\n- F1-Score: 0.7209\n- Accuracy score: 0.9992\n- AUC: 0.8443')
            # --- END Decision tree ---



            st.header('Compare the accuracy of the models on the Testing set')

            def compareResult():
                mylist=[]
                mylist2=[]

                mylist.append(acc1)
                mylist2.append("Logistic Regression")

                mylist.append(acc2)
                mylist2.append("Naive Bayes")

                mylist.append(acc4)
                mylist2.append("Decision Tree")


                plt.figure(figsize=(22, 10))
                sns.set_style("darkgrid")
                ax = sns.barplot(x = mylist2, y = mylist, palette = "Oranges", saturation =1.5)
                plt.xlabel("Classification Models", fontsize = 20 )
                plt.ylabel("Accuracy", fontsize = 20)
                plt.title("Accuracy of different Classification Models", fontsize = 20)
                plt.xticks(fontsize = 11, horizontalalignment = 'center', rotation = 0)
                plt.yticks(fontsize = 13)
                for p in ax.patches:
                    width, height = p.get_width(), p.get_height()
                    x, y = p.get_xy() 
                    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
                    
                plt.show()

            
            st.pyplot(compareResult())



            st.header('ROC Curve and Area Under the Curve')


            # Logistic Regression
            y_pred_logistic = LR_model.predict_proba(X_test)[:,1]
            logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred_logistic)
            auc_logistic = auc(logistic_fpr, logistic_tpr)


            # Naive Bayes
            y_pred_nb = NB_model.predict_proba(X_test)[:,1]
            nb_fpr, nb_tpr, threshold = roc_curve(y_test, y_pred_nb)
            auc_nb = auc(nb_fpr, nb_tpr)


            # Decision Tree
            y_pred_dtr = DTR_model.predict_proba(X_test)[:,1]
            dtr_fpr, dtr_tpr, threshold = roc_curve(y_test, y_pred_dtr)
            auc_dtr = auc(dtr_fpr, dtr_tpr)


            def plottingGraphResultCompare():
                plt.figure(figsize=(10, 8), dpi=100)
                plt.plot([0, 1], [0, 1], 'k--')
                # Logistic Regression
                plt.plot(logistic_fpr, logistic_tpr, label='Logistic Regression (auc = %0.4f)' % auc_logistic)
                # Naive Bayes
                plt.plot(nb_fpr, nb_tpr, label='Naive Bayes (auc = %0.4f)' % auc_nb)

                # Decision Tree
                plt.plot(dtr_fpr, dtr_tpr, label='Decision Tree (auc = %0.4f)' % auc_dtr)


                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')

                plt.legend(loc='best')
                plt.show()
            

            st.pyplot(plottingGraphResultCompare())
            import streamlit as st
            import shap
            from sklearn.linear_model import LogisticRegression
            import matplotlib.pyplot as plt

            # Train your machine learning model (e.g., Logistic Regression)
            logistic_model = LogisticRegression(random_state=0)
            logistic_model.fit(X_train, y_train)

            # Create an explainer object for the Logistic Regression model
            explainer = shap.Explainer(logistic_model, X_train)

            # Calculate SHAP values for the test set
            shap_values = explainer.shap_values(X_test)

            # Display SHAP summary plot
            summary_plot = shap.summary_plot(shap_values, X_test, plot_type="bar")
            st.pyplot(summary_plot)

            # Save SHAP summary plot as image
            summary_plot_fig = summary_plot[0].figure
            summary_plot_fig.savefig('shap_summary_plot.png')

            # Display SHAP force plot for a single prediction
            force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0, :])
            st.pyplot(force_plot)

            # Save SHAP force plot as image
            force_plot_html = force_plot.data[0]
            force_plot_html.savefig('shap_force_plot.png', format='png')

            # Display SHAP summary plot for the top features
            summary_plot_top_features = shap.summary_plot(shap_values, X_test)
            st.pyplot(summary_plot_top_features)

            # Save SHAP summary plot for the top features as image
            summary_plot_top_features_fig = summary_plot_top_features[0].figure
            summary_plot_top_features_fig.savefig('shap_summary_plot_top_features.png')

        # --- 4 CHECKBOX ---



    



    elif selected_menu == "Indicators":
        st.markdown(f"<h1 style='font-size: 24px;'>You have selected {selected_menu}</h1>", unsafe_allow_html=True)
    elif selected_menu == "Network":
        st.markdown(f"<h1 style='font-size: 24px;'>You have selected {selected_menu}</h1>", unsafe_allow_html=True)
        image_path = "/Users/nina/Desktop/Screenshot 2023-10-24 at 6.38.33 PM.png"

        # Display the local image
        st.image(image_path, caption='Your Image Caption', use_column_width=True)
    elif selected_menu == "Feedback":
        st.markdown(f"<h1 style='font-size: 24px;'>Was prediction correct?</h1>", unsafe_allow_html=True)
        from streamlit_feedback import streamlit_feedback
        feedback = streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="[Optional] Please provide an explanation",
        )
        feedback

