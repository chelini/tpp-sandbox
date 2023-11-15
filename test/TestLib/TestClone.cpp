#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
// This is a test pass for verifying mlir::clone.
struct TestCloneApi
    : public PassWrapper<TestCloneApi, InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCloneApi)

  void runOnOperation() override;
  StringRef getArgument() const final { return "test-clone"; }
  StringRef getDescription() const final { return "Test clone api."; }
};
} // namespace

struct DumpNotifications : public RewriterBase::Listener {
  void notifyOperationRemoved(Operation *op) override {
    llvm::outs() << "notifyOperationRemoved: " << op->getName() << "\n";
  }
  void notifyOperationInserted(Operation *op) override {
    llvm::outs() << "notifyOperationInserted: " << op->getName() << "\n";
  }
};

void TestCloneApi::runOnOperation() {
  MLIRContext *ctx = &getContext();
  mlir::RewritePatternSet patterns(ctx);

  DumpNotifications l = DumpNotifications();
  OpBuilder b(ctx, &l);

  getOperation().walk([&](linalg::GenericOp linalgOp) {
    b.setInsertionPoint(linalgOp);
    b.clone(*linalgOp.getOperation());
    llvm::outs() << "------\n";
    mlir::clone(b, linalgOp.getOperation(), linalgOp->getResults()[0].getType(),
                linalgOp->getOperands());
  });
}

namespace mlir {
namespace tpp {
void registerTestCloneApi() { PassRegistration<TestCloneApi>(); }
} // namespace tpp
} // namespace mlir
