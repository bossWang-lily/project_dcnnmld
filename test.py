from dcnn import *
import matplotlib.pyplot as plt


def generate_test_sets():
    sir_db = 10
    test_rho_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for test_rho in test_rho_list:
        print("Generating data sets, rho={:.1f} sir={}".format(test_rho, sir_db))
        test_set = DataSet(flag=2, rho=test_rho, sir=sir_db)
        test_set.produce_all()


def benchmark(k):
    sir_db = 10
    test_rho_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    sir_list = []
    rho_list = []

    ber_mld_list = []
    ber_baseline_list = []
    ber_improved_list = []

    mse_baseline_list = []
    mse_improved_list = []

    jb_baseline_list = []
    jb_improved_list = []

    for rho in test_rho_list:
        test_io = DataSet(flag=2, rho=rho, sir=sir_db)

        baseline_dcnnmld = DCNNMLD(rho, sir_db, is_improved=False)
        improved_dcnnmld = DCNNMLD(rho, sir_db, is_improved=True)

        baseline_dcnnmld.load()
        improved_dcnnmld.load()

        err_mld = 0
        err_baseline = 0
        err_improved = 0
        r_baseline = None
        r_improved = None
        total = 0

        idx = 0
        for y, h, s, one_hot, w, hat_s, hat_w in test_io.fetch():
            print("Testing rho={:.1f} sir={} batch={}/{}".format(rho, sir_db, idx + 1, TEST_TOTAL_BATCH), end="\r")

            bits = get_bits(s)
            total += bits.size

            bits_mld = get_bits(hat_s)
            err_mld += len(np.argwhere(bits_mld != bits))

            bits_baseline, w_baseline = baseline_dcnnmld.detect_bits_batch(y, h, hat_w, k)
            err_baseline += len(np.argwhere(bits_baseline != bits))
            r_baseline = concatenate(r_baseline, w - w_baseline)

            bits_improved, w_improved = improved_dcnnmld.detect_bits_batch(y, h, hat_w, k)
            err_improved += len(np.argwhere(bits_improved != bits))
            r_improved = concatenate(r_improved, w - w_improved)

            idx += 1
        print()

        baseline_dcnnmld.close()
        improved_dcnnmld.close()

        ber_mld = err_mld / total
        ber_baseline = err_baseline / total
        ber_improved = err_improved / total

        mse_baseline = np.mean(r_baseline ** 2)
        mse_improved = np.mean(r_improved ** 2)

        jb_baseline = jbtest(r_baseline)
        jb_improved = jbtest(r_improved)

        print("rho={:.1f}".format(rho))
        print("sir={}".format(sir_db))

        print("ber_mld={:e}".format(ber_mld))
        print("ber_baseline={:e}".format(ber_baseline))
        print("ber_improved={:e}".format(ber_improved))

        print("mse_baseline={:e}".format(mse_baseline))
        print("mse_improved={:e}".format(mse_improved))

        print("jbtest_baseline={:e}".format(jb_baseline))
        print("jbtest_improved={:e}".format(jb_improved))
        print()

        rho_list.append(rho)
        sir_list.append(sir_db)

        ber_mld_list.append(ber_mld)
        ber_baseline_list.append(ber_baseline)
        ber_improved_list.append(ber_improved)

        mse_baseline_list.append(mse_baseline)
        mse_improved_list.append(mse_improved)

        jb_baseline_list.append(jb_baseline)
        jb_improved_list.append(jb_improved)

    print("BENCHMARK RESULT")
    print("K={} NORMALIZED_DOPPLER_FREQUENCY={}".format(k, NORMALIZED_DOPPLER_FREQUENCY))
    print("rho\tsir_db\tber_mld\t\tber_baseline\tber_improved\tmse_baseline\tmse_improved\tjbtest_baseline\tjbtest_improved")
    for i in range(len(rho_list)):
        rho = rho_list[i]
        sir_db = sir_list[i]
        ber_mld = ber_mld_list[i]
        ber_baseline = ber_baseline_list[i]
        ber_improved = ber_improved_list[i]

        mse_improved = mse_improved_list[i]
        mse_baseline = mse_baseline_list[i]

        jb_baseline = jb_baseline_list[i]
        jb_improved = jb_improved_list[i]
        print("{:.1f}\t{}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}".format(
            rho,
            sir_db,
            ber_mld, ber_baseline, ber_improved,
            mse_baseline, mse_improved,
            jb_baseline, jb_improved))

    # 画个小图图
    plt.semilogy(rho_list, ber_mld_list)
    plt.semilogy(rho_list, ber_baseline_list)
    plt.semilogy(rho_list, ber_improved_list)
    plt.legend(["Standard MLD", "Baseline DCNN-MLD(K={})".format(k), "Improved DCNN-MLD(K={})".format(k)])
    plt.show()


if __name__ == "__main__":
    generate_test_sets()
    benchmark(k=1)
