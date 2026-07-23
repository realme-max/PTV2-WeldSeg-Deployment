#include "AppStateMachine.h"

#include <QtTest>

class AppStateMachineTest final : public QObject
{
    Q_OBJECT

private slots:
    void validWorkflow()
    {
        ptv2::qtui::AppStateMachine state;
        QString error;
        QVERIFY(state.transition(ptv2::qtui::AppState::kInitializing, error));
        QVERIFY(state.transition(ptv2::qtui::AppState::kReady, error));
        QVERIFY(state.transition(ptv2::qtui::AppState::kCloudSelected, error));
        QVERIFY(state.canDetect(true));
        QVERIFY(state.transition(ptv2::qtui::AppState::kDetecting, error));
        QVERIFY(!state.canDetect(true));
        QVERIFY(!state.canExport());
        QVERIFY(state.transition(ptv2::qtui::AppState::kDetectionSucceeded, error));
        QVERIFY(state.canExport());
        QVERIFY(state.transition(ptv2::qtui::AppState::kExporting, error));
        QVERIFY(!state.transition(ptv2::qtui::AppState::kShuttingDown, error)); // 5.
        QVERIFY(state.transition(ptv2::qtui::AppState::kDetectionSucceeded, error));
        QVERIFY(state.transition(ptv2::qtui::AppState::kShuttingDown, error));
    }

    void invalidTransitionsFailClosed()
    {
        ptv2::qtui::AppStateMachine state;
        QString error;
        QVERIFY(!state.transition(ptv2::qtui::AppState::kDetecting, error)); // 6.
        QVERIFY(!state.transition(ptv2::qtui::AppState::kExporting, error)); // 7.
        QVERIFY(!state.canDetect(false));
        QVERIFY(!state.canExport());
    }
};

QTEST_GUILESS_MAIN(AppStateMachineTest)
#include "AppStateMachineTest.moc"
