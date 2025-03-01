import streamlit as st
import pandas as pd
import time
import random
import psycopg2.extras
from psycopg2 import extras
from utils.database import init_connection

def show_annotation_page():
    st.title(f"Label Annotation")

    if 'show_label_info' not in st.session_state:
        st.session_state.show_label_info = False

    df = st.session_state.df
    index = st.session_state.index

    if index < len(df):
        row = df.iloc[index]

        if index not in st.session_state.percent_confident:
            st.session_state.percent_confident[index] = random.randint(0, 100)

        percent_confident = st.session_state.percent_confident[index]

        # Assign a random label instead of using row['esci_label']
        random_label = random.choice(["E", "S", "C", "I"])

        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("#### **Query**")
            st.write(f"{row['query']}")
            st.write("#### **Product**")
            st.write(f"{row['product']}")

        with col2:
            st.write(f"#### **Current Label**: ")
            st.markdown(f"<h2 style='text-align: center; color: black;'>{random_label}</h2>", unsafe_allow_html=True)
            st.write(f"#### **Confidence**: {percent_confident}%")

        st.write("---")

        if 'temp_selection' not in st.session_state:
            st.session_state['temp_selection'] = None

        if st.button("Confirm Current Label", key='confirm_current', use_container_width=True):
            save_annotation(row, random_label, percent_confident)

        selected_label = st.selectbox(
            "Choose a new label:",
            options=["Exact (E)", "Substitute (S)", "Complementary (C)", "Irrelevant (I)"],
            index=None,
            placeholder="Select a label..."
        )

        if selected_label:
            st.info(f"Current selection: {selected_label}")

        if st.button("Confirm New Label", key='confirm_new', use_container_width=True):
            if selected_label:
                label_map = {
                    "Exact (E)": "E",
                    "Substitute (S)": "S",
                    "Complementary (C)": "C",
                    "Irrelevant (I)": "I"
                }
                save_annotation(row, label_map[selected_label], percent_confident)
            else:
                st.warning("Please select a label before confirming.")

        st.write("---")

        if len(st.session_state.annotations) > 0:
            st.write("### Golden Dataset")
            annotated_df = pd.DataFrame(st.session_state.annotations)
            st.dataframe(annotated_df)

            if st.button("Push Annotations to Database", key='push_annotations', use_container_width=True):
                push_annotations_to_database(annotated_df)
                st.success("Annotations successfully pushed to database!")
        else:
            st.write("### No Annotations Yet")
        
        # Define the label data
        label_data = pd.DataFrame({
            "Label": ["Exact (E)", "Substitute (S)", "Complementary (C)", "Irrelevant (I)"],
            "Definition": [
                "The product is an exact match to the query.",
                "The product does not fully match the query but can serve as a functional alternative.",
                "The product does not directly fulfill the query but can be used in conjunction with the query.",
                "The product does not relate to the query or fails to fulfill its central aspects."
            ],
            "Example": [
                "Query: iPhone<br>Product: iPhone ",
                "Query: iPhone<br>Product: Samsung Phone",
                "Query: iPhone<br>Product: iPhone Charger",
                "Query: iPhone<br>Product: Banana"
            ]
        })

        # add spacing 
        st.write("---")
        st.write("")  #

        # buttons at the bottom
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("Back to Login Page", key='back_to_main', use_container_width=True):
                st.session_state.clear()
                st.session_state.page = 'login'
                st.rerun()

        with col2:
            if len(st.session_state.annotation_history) > 0:
                if st.button("Redo Previous Annotation", key='redo_previous', use_container_width=True):
                    if len(st.session_state.annotations) > 0:
                        st.session_state.annotations.pop()
                    st.session_state.annotation_history.pop()
                    if st.session_state.index > 0:
                        st.session_state.index -= 1
                    st.rerun()

        with col3:
            if st.button("Show/Hide Label Definitions", key='toggle_labels', use_container_width=True):
                st.session_state.show_label_info = not st.session_state.show_label_info

        # show label definitions if toggled
        if st.session_state.show_label_info:
            st.write("### Label Definitions and Examples")
            st.markdown(label_data.to_html(escape=False, index=False), unsafe_allow_html=True)

def push_annotations_to_database(annotations_df):
    # get the username and current timestamp
    user_id = st.session_state.username
    current_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')  

    conn = init_connection()
    if conn is None:
        st.error("Database connection failed.")
        return

    try:
        with conn.cursor() as cur:
            # get analystID based on user_id
            cur.execute("""
                SELECT "analystID" FROM tbl_analyst WHERE "username" = %s;
            """, (user_id,))
            analyst_row = cur.fetchone()
            
            if analyst_row is None:
                st.error("Analyst ID not found for the current user.")
                return
            
            analyst_id = analyst_row[0]

            # insert a new golden version
            cur.execute(
                """
                INSERT INTO tbl_versiongolden ("goldenVersion")
                VALUES (%s)
                RETURNING "goldenID";
                """,
                (current_timestamp,)
            )
            latest_golden_id = cur.fetchone()[0]

            if latest_golden_id is None:
                st.error("Failed to retrieve latest goldenID.")
                return

            # convert the rows into tuples for SQL execution
            annotations_tuples = [
                (row["query"], row["product_title"], row["annotated_label"])
                for _, row in annotations_df.iterrows()
            ]

            # temporary table for bulk insert
            with conn.cursor() as tmp_cur:
                tmp_cur.execute("""
                CREATE TEMP TABLE tmp_import (
                    "query" TEXT, 
                    "product_title" TEXT, 
                    "esci_label" TEXT
                ) ON COMMIT DROP;
                """)
                
                psycopg2.extras.execute_values(
                    tmp_cur,
                    """INSERT INTO tmp_import ("query", "product_title", "esci_label") 
                       VALUES %s""",
                    annotations_tuples
                )

                # insert query and product into tbl_queryproducts
                tmp_cur.execute("""
                INSERT INTO tbl_queryproducts ("query", "product")
                SELECT "query", "product_title"
                FROM tmp_import
                ON CONFLICT DO NOTHING;
                """)

                # insert esci label IDs paired with qpID and analystID into tbl_golden
                tmp_cur.execute("""
                WITH latest_golden AS (
                   SELECT MAX("goldenID") AS "goldenID" FROM tbl_versiongolden
                )
                INSERT INTO tbl_golden ("qpID", "esciID", "goldenID", "analystID")
                SELECT DISTINCT ON (qp."qpID", lg."goldenID")
                   qp."qpID", 
                   e."esciID",
                   lg."goldenID",
                   %s  -- Adding analystID here
                FROM tmp_import t
                JOIN tbl_queryproducts qp 
                   ON t."query" = qp."query" AND t."product_title" = qp."product"
                JOIN tbl_esci e 
                   ON t."esci_label" = e."esciLabel"
                CROSS JOIN latest_golden lg
                ON CONFLICT ("qpID", "goldenID") 
                DO UPDATE SET 
                   "esciID" = EXCLUDED."esciID",
                   "analystID" = EXCLUDED."analystID";
                """, (analyst_id,))

                conn.commit()
                st.success("Annotations successfully pushed to database!")

    except Exception as e:
        st.error(f"Error inserting rows: {e}")
        conn.rollback()

    finally:
        conn.close()


def save_annotation(row, label, confidence):

    new_annotation = {
        'query': row['query'],
        'product_title': row['product'],
        'original_label': label,
        'annotated_label': label,
        'confidenceScore': confidence,
    }

    st.session_state.annotation_history.append(new_annotation)
    st.session_state.annotations.append(new_annotation)

    st.session_state.previous_index = st.session_state.index
    st.session_state.index += 1
    st.rerun()