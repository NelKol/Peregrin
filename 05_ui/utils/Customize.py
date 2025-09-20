from shiny import ui


class Format:

    Accordion = """
        #sidebar, #sidebar > *, #sidebar > div, #sidebar .accordion,
        #sidebar .accordion .accordion-header {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }
        #sidebar .accordion .accordion-item .accordion-header .accordion-button {
            font-size: 0.6em !important;
            /* font-weight: bold !important; */
            padding-left: 10px !important;
            padding-right: 0 !important;
        }
        #sidebar .accordion .accordion-collapse {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }
        #sidebar .accordion .accordion-body {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            box-sizing: border-box !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        #sidebar .accordion .accordion-item {
            border-left: 0 !important;
            border-right: 0 !important;
        }
    """

