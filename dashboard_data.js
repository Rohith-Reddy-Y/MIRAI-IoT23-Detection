// IoT-23 Dashboard Data — All 23 datasets
const DATASETS = [
    { id: "dataset1", status: "single_class_train", features: 16, trainRows: 316, valRows: 68, testRows: 68, classTrain: { 0: 316 }, classVal: null, classTest: null, test: null, val: null, train: null },
    { id: "dataset2", status: "single_class_train", features: 18, trainRows: 964, valRows: 204, testRows: 206, classTrain: { 0: 964 }, classVal: null, classTest: null, test: null, val: null, train: null },
    { id: "dataset3", status: "single_class_train", features: 17, trainRows: 91, valRows: 19, testRows: 20, classTrain: { 0: 91 }, classVal: null, classTest: null, test: null, val: null, train: null },
    {
        id: "dataset4", status: "trained", features: 16, trainRows: 4464816, valRows: 956765, testRows: 956712,
        classTrain: { 0: 17248, 1: 4447568 }, classVal: { 0: 2882, 1: 953883 }, classTest: { 0: 2418, 1: 954294 },
        scalePos: 0.00388, threshold: 0.768,
        val: { rocAuc: 1.0, prAuc: 1.0, f1: 1.0, acc: 1.0, cm: { tn: 2882, fp: 0, fn: 0, tp: 953883 } },
        test: { rocAuc: 0.9994, prAuc: 1.0, acc: 0.99999, cm: { tn: 2415, fp: 3, fn: 0, tp: 954294 } },
        train: { rocAuc: 0.9999, prAuc: 1.0, acc: 0.99999, cm: { tn: 17243, fp: 5, fn: 0, tp: 4447568 } }
    },
    {
        id: "dataset5", status: "trained", features: 13, trainRows: 7282, valRows: 1560, testRows: 1561,
        classTrain: { 0: 1536, 1: 5746 }, classVal: { 0: 347, 1: 1213 }, classTest: { 0: 298, 1: 1263 },
        scalePos: 0.267, threshold: 0.990,
        val: { rocAuc: 1.0, prAuc: 1.0, f1: 1.0, acc: 1.0, cm: { tn: 347, fp: 0, fn: 0, tp: 1213 } },
        test: { rocAuc: 1.0, prAuc: 1.0, acc: 1.0, cm: { tn: 298, fp: 0, fn: 0, tp: 1263 } },
        train: { rocAuc: 1.0, prAuc: 1.0, acc: 0.9997, cm: { tn: 1534, fp: 2, fn: 0, tp: 5746 } }
    },
    {
        id: "dataset6", status: "trained", features: 16, trainRows: 8018300, valRows: 1718216, testRows: 1718198,
        classTrain: { 0: 27139, 1: 7991161 }, classVal: { 0: 1259, 1: 1716957 }, classTest: { 0: 47557, 1: 1670641 },
        scalePos: 0.0034, threshold: 0.019,
        val: { rocAuc: 0.9991, prAuc: 1.0, f1: 0.99999, acc: 0.99999, cm: { tn: 1248, fp: 11, fn: 6, tp: 1716951 } },
        test: { rocAuc: 0.5113, prAuc: 0.9729, acc: 0.9731, cm: { tn: 1369, fp: 46188, fn: 11, tp: 1670630 } },
        train: { rocAuc: 0.9996, prAuc: 1.0, acc: 0.99999, cm: { tn: 27075, fp: 64, fn: 26, tp: 7991135 } }
    },
    {
        id: "dataset7", status: "trained", features: 15, trainRows: 2506741, valRows: 537186, testRows: 537101,
        classTrain: { 0: 1835, 1: 2504906 }, classVal: { 0: 312, 1: 536874 }, classTest: { 0: 329, 1: 536772 },
        scalePos: 0.00073, threshold: 0.570,
        val: { rocAuc: 0.5, prAuc: 0.9994, f1: 0.9997, acc: 0.9994, cm: { tn: 0, fp: 312, fn: 0, tp: 536874 } },
        test: { rocAuc: 0.5745, prAuc: 0.9995, acc: 0.9995, cm: { tn: 49, fp: 280, fn: 6, tp: 536766 } },
        train: { rocAuc: 0.6237, prAuc: 0.9994, acc: 0.9994, cm: { tn: 454, fp: 1381, fn: 74, tp: 2504832 } }
    },
    {
        id: "dataset8", status: "trained", features: 17, trainRows: 13846974, valRows: 2967213, testRows: 2967191,
        classTrain: { 0: 1239, 1: 13845735 }, classVal: { 0: 290, 1: 2966923 }, classTest: { 0: 265, 1: 2966926 },
        scalePos: 0.0000895, threshold: 0.051,
        val: { rocAuc: 0.9942, prAuc: 1.0, f1: 0.99999, acc: 0.9971, cm: { tn: 282, fp: 8, fn: 8573, tp: 2958350 } },
        test: { rocAuc: 0.8615, prAuc: 1.0, acc: 0.9964, cm: { tn: 257, fp: 8, fn: 10815, tp: 2956111 } },
        train: { rocAuc: 0.9974, prAuc: 1.0, acc: 0.9959, cm: { tn: 1203, fp: 36, fn: 56135, tp: 13789600 } }
    },
    {
        id: "dataset9", status: "trained", features: 17, trainRows: 3787395, valRows: 811583, testRows: 811583,
        classTrain: { 0: 2668, 1: 3784727 }, classVal: { 0: 482, 1: 811101 }, classTest: { 0: 515, 1: 811068 },
        scalePos: 0.000705, threshold: 0.003,
        val: { rocAuc: 0.9999, prAuc: 1.0, f1: 0.99996, acc: 0.9998, cm: { tn: 481, fp: 1, fn: 192, tp: 810909 } },
        test: { rocAuc: 0.9989, prAuc: 1.0, acc: 0.9998, cm: { tn: 514, fp: 1, fn: 178, tp: 810890 } },
        train: { rocAuc: 0.9993, prAuc: 1.0, acc: 0.9998, cm: { tn: 2657, fp: 11, fn: 877, tp: 3783850 } }
    },
    {
        id: "dataset10", status: "trained", features: 16, trainRows: 2376036, valRows: 509151, testRows: 509151,
        classTrain: { 0: 2734, 1: 2373302 }, classVal: { 0: 502, 1: 508649 }, classTest: { 0: 498, 1: 508653 },
        scalePos: 0.00115, threshold: 0.084,
        val: { rocAuc: 0.9985, prAuc: 1.0, f1: 0.99999, acc: 0.99999, cm: { tn: 501, fp: 1, fn: 3, tp: 508646 } },
        test: { rocAuc: 0.9999, prAuc: 1.0, acc: 0.99999, cm: { tn: 497, fp: 1, fn: 2, tp: 508651 } },
        train: { rocAuc: 0.99999, prAuc: 1.0, acc: 0.99999, cm: { tn: 2732, fp: 2, fn: 24, tp: 2373278 } }
    },
    {
        id: "dataset11", status: "trained", features: 14, trainRows: 166, valRows: 35, testRows: 36,
        classTrain: { 0: 141, 1: 25 }, classVal: { 0: 35, 1: 0 }, classTest: { 0: 35, 1: 1 },
        scalePos: 5.64, threshold: 0.5,
        val: { rocAuc: null, prAuc: null, f1: null, acc: 1.0, note: "single_class", cm: null },
        test: { rocAuc: 1.0, prAuc: 1.0, acc: 0.9722, cm: { tn: 35, fp: 0, fn: 1, tp: 0 } },
        train: { rocAuc: 1.0, prAuc: 1.0, acc: 0.9880, cm: { tn: 139, fp: 2, fn: 0, tp: 25 } }
    },
    {
        id: "dataset12", status: "trained", features: 16, trainRows: 47125276, valRows: 10098268, testRows: 10098265,
        classTrain: { 0: 14398173, 1: 32727103 }, classVal: { 0: 3088165, 1: 7010103 }, classTest: { 0: 3088596, 1: 7009669 },
        scalePos: 0.4399, threshold: 0.233,
        val: { rocAuc: 0.4997, prAuc: 0.6941, f1: 0.8195, acc: 0.3285, cm: { tn: 2905203, fp: 182962, fn: 6597809, tp: 412294 } },
        test: { rocAuc: 0.5002, prAuc: 0.6943, acc: 0.3286, cm: { tn: 2909553, fp: 179043, fn: 6600902, tp: 408767 } },
        train: { rocAuc: 0.5017, prAuc: 0.6958, acc: 0.3380, cm: { tn: 13252212, fp: 1145961, fn: 30050631, tp: 2676472 } }
    },
    {
        id: "dataset13", status: "trained", features: 17, trainRows: 3098, valRows: 665, testRows: 663,
        classTrain: { 0: 3092, 1: 6 }, classVal: { 0: 665, 1: 0 }, classTest: { 0: 663, 1: 0 },
        scalePos: 515.33, threshold: 0.5,
        val: { rocAuc: null, prAuc: null, f1: null, acc: 1.0, note: "single_class", cm: null },
        test: { rocAuc: null, prAuc: null, acc: 1.0, note: "single_class", cm: null },
        train: { rocAuc: 1.0, prAuc: 1.0, acc: 1.0, cm: { tn: 3092, fp: 0, fn: 0, tp: 6 } }
    },
    {
        id: "dataset14", status: "trained", features: 19, trainRows: 51498290, valRows: 11035354, testRows: 11035337,
        classTrain: { 0: 5190, 1: 51493100 }, classVal: { 0: 1152, 1: 11034202 }, classTest: { 0: 995, 1: 11034342 },
        scalePos: 0.000101, threshold: 0.00144,
        val: { rocAuc: 0.99999, prAuc: 1.0, f1: 0.99999, acc: 0.99997, cm: { tn: 1152, fp: 0, fn: 345, tp: 11033857 } },
        test: { rocAuc: 0.6724, prAuc: 0.9999, acc: 0.99997, cm: { tn: 995, fp: 0, fn: 291, tp: 11034051 } },
        train: { rocAuc: 0.99999, prAuc: 1.0, acc: 0.99997, cm: { tn: 5190, fp: 0, fn: 1438, tp: 51491662 } }
    },
    {
        id: "dataset15", status: "trained", features: 16, trainRows: 9551569, valRows: 2046767, testRows: 2046762,
        classTrain: { 0: 1604, 1: 9549965 }, classVal: { 0: 425, 1: 2046342 }, classTest: { 0: 634, 1: 2046128 },
        scalePos: 0.000168, threshold: 0.452,
        val: { rocAuc: 1.0, prAuc: 1.0, f1: 1.0, acc: 0.9988, cm: { tn: 425, fp: 0, fn: 2361, tp: 2043981 } },
        test: { rocAuc: 0.4984, prAuc: 0.9997, acc: 0.9988, cm: { tn: 634, fp: 0, fn: 2364, tp: 2043764 } },
        train: { rocAuc: 1.0, prAuc: 1.0, acc: 0.9988, cm: { tn: 1604, fp: 0, fn: 11030, tp: 9538935 } }
    },
    {
        id: "dataset16", status: "trained", features: 18, trainRows: 7313451, valRows: 1567183, testRows: 1567153,
        classTrain: { 0: 5465860, 1: 1847591 }, classVal: { 0: 1567009, 1: 174 }, classTest: { 0: 1229520, 1: 337633 },
        scalePos: 0.338, threshold: 0.444,
        val: null,
        test: { rocAuc: 0.3612, prAuc: 0.2410, acc: 0.7846, cm: null },
        train: null
    },
    {
        id: "dataset17", status: "trained", features: 17, trainRows: 16209, valRows: 3476, testRows: 3460,
        classTrain: { 0: 1753, 1: 14456 }, classVal: { 0: 0, 1: 3476 }, classTest: { 0: 170, 1: 3290 },
        scalePos: 0.121, threshold: 0.5,
        val: { rocAuc: null, prAuc: null, f1: null, acc: 1.0, note: "single_class", cm: null },
        test: { rocAuc: 1.0, prAuc: 1.0, acc: 0.9616, cm: { tn: 170, fp: 0, fn: 133, tp: 3157 } },
        train: { rocAuc: 0.9976, prAuc: 0.9996, acc: 0.9892, cm: { tn: 1700, fp: 53, fn: 122, tp: 14334 } }
    },
    {
        id: "dataset18", status: "trained", features: 17, trainRows: 38118220, valRows: 8168187, testRows: 8168184,
        classTrain: { 0: 966846, 1: 37151374 }, classVal: { 0: 207006, 1: 7961181 }, classTest: { 0: 206939, 1: 7961245 },
        scalePos: 0.026, threshold: 0.018,
        val: { rocAuc: 0.5077, prAuc: 0.975, f1: 0.9873, acc: 0.9750, cm: { tn: 3196, fp: 203810, fn: 669, tp: 7960512 } },
        test: { rocAuc: 0.5019, prAuc: 0.9748, acc: 0.9750, cm: { tn: 3053, fp: 203886, fn: 577, tp: 7960668 } },
        train: { rocAuc: 0.5082, prAuc: 0.975, acc: 0.9750, cm: { tn: 15789, fp: 951057, fn: 2909, tp: 37148465 } }
    },
    {
        id: "dataset19", status: "trained", features: 18, trainRows: 109272, valRows: 23415, testRows: 23416,
        classTrain: { 0: 3728, 1: 105544 }, classVal: { 0: 287, 1: 23128 }, classTest: { 0: 521, 1: 22895 },
        scalePos: 0.0353, threshold: 0.241,
        val: { rocAuc: 1.0, prAuc: 1.0, f1: 1.0, acc: 0.9991, cm: { tn: 287, fp: 0, fn: 22, tp: 23106 } },
        test: { rocAuc: 0.9996, prAuc: 1.0, acc: 0.9963, cm: { tn: 505, fp: 16, fn: 71, tp: 22824 } },
        train: { rocAuc: 0.99996, prAuc: 1.0, acc: 0.9980, cm: { tn: 3719, fp: 9, fn: 207, tp: 105337 } }
    },
    {
        id: "dataset20", status: "trained", features: 15, trainRows: 2300, valRows: 493, testRows: 493,
        classTrain: { 0: 2287, 1: 13 }, classVal: { 0: 493, 1: 0 }, classTest: { 0: 492, 1: 1 },
        scalePos: 175.92, threshold: 0.5,
        val: { rocAuc: null, prAuc: null, f1: null, acc: 1.0, note: "single_class", cm: null },
        test: { rocAuc: 1.0, prAuc: 1.0, acc: 1.0, cm: { tn: 492, fp: 0, fn: 0, tp: 1 } },
        train: { rocAuc: 0.99997, prAuc: 0.989, acc: 0.99957, cm: { tn: 2286, fp: 1, fn: 0, tp: 13 } }
    },
    {
        id: "dataset21", status: "trained", features: 14, trainRows: 2246, valRows: 481, testRows: 482,
        classTrain: { 0: 2230, 1: 16 }, classVal: { 0: 481, 1: 0 }, classTest: { 0: 482, 1: 0 },
        scalePos: 139.375, threshold: 0.5,
        val: { rocAuc: null, prAuc: null, f1: null, acc: 1.0, note: "single_class", cm: null },
        test: { rocAuc: null, prAuc: null, acc: 1.0, note: "single_class", cm: null },
        train: { rocAuc: 0.99999, prAuc: 0.9963, acc: 0.99955, cm: { tn: 2229, fp: 1, fn: 0, tp: 16 } }
    },
    {
        id: "dataset22", status: "trained", features: 16, trainRows: 38261904, valRows: 8198972, testRows: 8198979,
        classTrain: { 0: 22080, 1: 38239824 }, classVal: { 0: 4637, 1: 8194335 }, classTest: { 0: 4721, 1: 8194258 },
        scalePos: 0.000577, threshold: 0.000561,
        val: { rocAuc: 0.9867, prAuc: 0.99999, f1: 0.9999, acc: 0.9931, cm: { tn: 4100, fp: 537, fn: 56402, tp: 8137933 } },
        test: { rocAuc: 0.5266, prAuc: 0.9994, acc: 0.9929, cm: { tn: 4140, fp: 581, fn: 57336, tp: 8136922 } },
        train: { rocAuc: 0.9888, prAuc: 0.99999, acc: 0.9930, cm: { tn: 19723, fp: 2357, fn: 264218, tp: 37975606 } }
    },
    {
        id: "dataset23", status: "trained", features: 18, trainRows: 706124, valRows: 151312, testRows: 151312,
        classTrain: { 0: 324990, 1: 381134 }, classVal: { 0: 72144, 1: 79168 }, classTest: { 0: 72141, 1: 79171 },
        scalePos: 0.853, threshold: 0.897,
        val: { rocAuc: 0.9550, prAuc: 0.9242, f1: 0.9603, acc: 0.9568, cm: { tn: 65611, fp: 6533, fn: 4, tp: 79164 } },
        test: { rocAuc: 0.9548, prAuc: 0.9239, acc: 0.9568, cm: { tn: 65604, fp: 6537, fn: 7, tp: 79164 } },
        train: { rocAuc: 0.9556, prAuc: 0.9332, acc: 0.9569, cm: { tn: 294571, fp: 30419, fn: 38, tp: 381096 } }
    }
];

const FEATURES = ["duration", "local_orig", "local_resp", "missed_bytes", "orig_bytes", "orig_ip_bytes", "orig_pkts", "proto_icmp", "proto_tcp", "proto_udp", "resp_bytes", "resp_ip_bytes", "resp_pkts", "service_-", "service_dns", "service_http", "service_irc", "service_ssh", "service_ssl"];

const PIPELINE_STEPS = [
    { icon: "📥", title: "Raw CSV", desc: "23 IoT-23 network traffic datasets" },
    { icon: "🧹", title: "Data Cleaning", desc: "Handle missing values, parse labels" },
    { icon: "🔢", title: "Encoding", desc: "One-hot encode categorical features" },
    { icon: "📏", title: "Normalization", desc: "MinMax scaling of numeric features" },
    { icon: "✂️", title: "Train/Val/Test Split", desc: "70/15/15 stratified splitting" },
    { icon: "🎯", title: "Feature Selection", desc: "XGBoost gain + correlation fallback" },
    { icon: "🤖", title: "Stage 1: Binary Training", desc: "XGBoost binary: Benign vs Malicious" },
    { icon: "🧬", title: "Stage 2: Multiclass Training", desc: "XGBoost multi:softprob for attack variants" },
    { icon: "📊", title: "Evaluation", desc: "ROC-AUC, F1, Accuracy, Confusion Matrix" }
];

// Multiclass model performance — ALL 23 datasets (from master_summary_multiclass.json)
const MULTICLASS_STATS = [
    { id: "dataset1", accuracy: 1.0, f1_weighted: null, f1_macro: null, classes: 1, testClasses: 1, variants: ["-"] },
    { id: "dataset2", accuracy: 1.0, f1_weighted: null, f1_macro: null, classes: 1, testClasses: 1, variants: ["-"] },
    { id: "dataset3", accuracy: 1.0, f1_weighted: null, f1_macro: null, classes: 1, testClasses: 1, variants: ["-"] },
    { id: "dataset4", accuracy: 0.9999969, f1_weighted: 0.9999969, f1_macro: 0.9996888, classes: 2, testClasses: 2, variants: ["-", "PartOfAHorizontalPortScan"] },
    { id: "dataset5", accuracy: 1.0, f1_weighted: 1.0, f1_macro: 1.0, classes: 2, testClasses: 2, variants: ["-", "PartOfAHorizontalPortScan"] },
    { id: "dataset6", accuracy: 0.9722779, f1_weighted: 0.9719614, f1_macro: 0.3390549, classes: 4, testClasses: 3, variants: ["-", "C&C", "C&C-HeartBeat", "DDoS"] },
    { id: "dataset7", accuracy: 0.9993875, f1_weighted: 0.9990813, f1_macro: 0.4998468, classes: 3, testClasses: 2, variants: ["-", "C&C-HeartBeat", "DDoS"] },
    { id: "dataset8", accuracy: 0.9999124, f1_weighted: 0.9998704, f1_macro: 0.3456645, classes: 5, testClasses: 2, variants: ["-", "C&C", "C&C-HeartBeat", "DDoS", "PartOfAHorizontalPortScan"] },
    { id: "dataset9", accuracy: 0.9994184, f1_weighted: 0.9994887, f1_macro: 0.8657264, classes: 5, testClasses: 3, variants: ["-", "C&C", "C&C-HeartBeat", "DDoS", "PartOfAHorizontalPortScan"] },
    { id: "dataset10", accuracy: 0.9985623, f1_weighted: 0.9979718, f1_macro: 0.5324608, classes: 6, testClasses: 5, variants: ["-", "Attack", "C&C", "C&C-HeartBeat", "DDoS", "PartOfAHorizontalPortScan"] },
    { id: "dataset11", accuracy: 0.9722222, f1_weighted: 0.9585290, f1_macro: 0.4929577, classes: 2, testClasses: 2, variants: ["-", "C&C"] },
    { id: "dataset12", accuracy: 0.3058541, f1_weighted: 0.1432729, f1_macro: 0.1171088, classes: 7, testClasses: 4, variants: ["-", "Attack", "C&C", "C&C-FileDownload", "C&C-HeartBeat", "DDoS", "PartOfAHorizontalPortScan"] },
    { id: "dataset13", accuracy: 1.0, f1_weighted: 1.0, f1_macro: 1.0, classes: 3, testClasses: 1, variants: ["-", "C&C-HeartBeat", "PartOfAHorizontalPortScan"] },
    { id: "dataset14", accuracy: 0.0000517, f1_weighted: 0.0000486, f1_macro: 0.3546407, classes: 4, testClasses: 4, variants: ["-", "C&C", "C&C-HeartBeat", "PartOfAHorizontalPortScan"] },
    { id: "dataset15", accuracy: 0.9992764, f1_weighted: 0.9991310, f1_macro: 0.6290626, classes: 4, testClasses: 3, variants: ["-", "C&C", "DDoS", "PartOfAHorizontalPortScan"] },
    { id: "dataset16", accuracy: 0.7845533, f1_weighted: 0.6898380, f1_macro: 0.2930904, classes: 2, testClasses: 3, variants: ["-", "C&C-Torii"] },
    { id: "dataset17", accuracy: 0.3699422, f1_weighted: 0.4457264, f1_macro: 0.1775183, classes: 3, testClasses: 3, variants: ["-", "C&C", "DDoS"] },
    { id: "dataset18", accuracy: 0.0253562, f1_weighted: 0.0013073, f1_macro: 0.1554756, classes: 4, testClasses: 4, variants: ["-", "C&C", "C&C-HeartBeat", "DDoS"] },
    { id: "dataset19", accuracy: 0.0648702, f1_weighted: 0.0439702, f1_macro: 0.2589827, classes: 6, testClasses: 3, variants: ["-", "C&C", "C&C-HeartBeat", "C&C-Mirai", "DDoS", "PartOfAHorizontalPortScan"] },
    { id: "dataset20", accuracy: 0.9979716, f1_weighted: 0.9969584, f1_macro: 0.4994924, classes: 2, testClasses: 2, variants: ["-", "Okiru"] },
    { id: "dataset21", accuracy: 0.9937759, f1_weighted: 0.9968783, f1_macro: 0.4984391, classes: 2, testClasses: 1, variants: ["-", "Okiru"] },
    { id: "dataset22", accuracy: 0.0006009, f1_weighted: 0.0000635, f1_macro: 0.1009530, classes: 7, testClasses: 5, variants: ["-", "Attack", "C&C-HeartBeat", "C&C-HeartBeat-Attack", "DDoS", "Okiru", "PartOfAHorizontalPortScan"] },
    { id: "dataset23", accuracy: 0.4767699, f1_weighted: 0.3078469, f1_macro: 0.2152309, classes: 7, testClasses: 3, variants: ["-", "C&C", "C&C-FileDownload", "C&C-HeartBeat", "DDoS", "FileDownload", "PartOfAHorizontalPortScan"] }
];

const VARIANT_NAMES = {
    "-": "Benign",
    "Attack": "Attack",
    "C&C": "C&C Communication",
    "C&C-FileDownload": "C&C File Download",
    "C&C-HeartBeat": "C&C Heartbeat",
    "C&C-HeartBeat-Attack": "C&C Heartbeat Attack",
    "C&C-Mirai": "Mirai C&C",
    "C&C-Torii": "Torii C&C",
    "DDoS": "DDoS Attack",
    "FileDownload": "File Download",
    "Okiru": "Okiru Malware",
    "Okiru-Attack": "Okiru Attack",
    "PartOfAHorizontalPortScan": "Horizontal Port Scan",
    "PartOfAHorizontalPortScan-Attack": "Port Scan Attack"
};
