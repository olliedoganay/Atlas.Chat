const USER_CANCELLED_RUN_ERROR = /^run stopped by user\.?$/i;

export function isUserCancelledRunError(value: unknown) {
  return USER_CANCELLED_RUN_ERROR.test(String(value ?? "").trim());
}
