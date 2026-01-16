// I will rewrite the entire Login component with your requested changes:
// - "Don't have an account? Create one for free" moved to the RIGHT SIDE, below login container
// - Left side made more visually appealing but same theme

import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { useToast } from "@/hooks/use-toast";
import { loginCredentialsSchema, type LoginCredentials } from "@shared/schema";
import { apiRequest } from "@/lib/queryClient";
import { setSessionId } from "@/lib/auth";
import { Mail, Lock, ArrowRight, Eye, EyeOff, Sparkles } from "lucide-react";
import Navigation from "@/components/Navigation";

export default function Login() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const [rememberMe, setRememberMe] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  useEffect(() => {
    if (localStorage.getItem("isAuthenticated") === "true") {
      setLocation("/upload");
    }
  }, [setLocation]);

  const form = useForm<LoginCredentials>({
    resolver: zodResolver(loginCredentialsSchema),
    defaultValues: {
      username: "",
      password: "",
    },
  });

  const loginMutation = useMutation({
    mutationFn: async (credentials: LoginCredentials) => {
      const response = await apiRequest("POST", "/api/login", credentials);
      return (await response.json()) as { success: boolean; message: string };
    },
    onSuccess: (data) => {
      if (data.success) {
        try {
          const maybeSessionId = (data as any).sessionId || (data as any).session?.sessionId;
          if (maybeSessionId) setSessionId(maybeSessionId);
          const username = (data as any).user?.username || (data as any).username;
          if (username) localStorage.setItem("username", username);
        } catch (e) {}
        localStorage.setItem("isAuthenticated", "true");
        toast({
          title: "Welcome back!",
          description: "You've successfully signed in.",
        });
        setTimeout(() => {
          setLocation("/upload");
          window.dispatchEvent(new Event("auth-changed"));
        }, 1000);
      }
    },
    onError: (error: any) => {
      toast({
        title: "Sign in failed",
        description: error.message || "Please check your credentials and try again.",
        variant: "destructive",
      });
    },
  });

  const onSubmit = (data: LoginCredentials) => {
    loginMutation.mutate(data);
  };

  return (
    <div className="min-h-screen relative dark overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-blue-950/50 to-slate-950">
        <div className="absolute inset-0 opacity-30">
          <div
            className="absolute top-0 left-0 w-full h-full"
            style={{
              backgroundImage:
                "radial-gradient(circle at 25% 25%, rgba(59, 130, 246, 0.15) 0%, transparent 50%), radial-gradient(circle at 75% 75%, rgba(6, 182, 212, 0.15) 0%, transparent 50%)",
            }}
          />
        </div>
        <div className="absolute top-1/4 left-1/4 w-[550px] h-[550px] bg-blue-500/20 rounded-full blur-3xl animate-pulse" />
        <div
          className="absolute bottom-1/4 right-1/4 w-[450px] h-[450px] bg-cyan-500/20 rounded-full blur-3xl animate-pulse"
          style={{ animationDelay: "1s" }}
        />
      </div>

      <Navigation />

      <div className="relative z-10 min-h-screen flex items-center justify-center p-4 pt-20">
        <div className="w-full max-w-7xl mx-auto grid lg:grid-cols-2 gap-16 items-center">
          {/* LEFT SIDE – MORE APPEALING */}
          <motion.div
            className="hidden lg:flex flex-col items-start space-y-10 pl-4"
            initial={{ opacity: 0, x: -40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7 }}
          >
            <div className="space-y-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                <div className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-sm font-medium backdrop-blur-md shadow-lg shadow-blue-500/5">
                  <Sparkles className="w-4 h-4" /> Login Portal
                </div>

                <h1 className="text-6xl font-extrabold text-white leading-tight drop-shadow-xl">
                  Welcome Back
                  <span className="block bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-500 bg-clip-text text-transparent">
                    Access Your Account
                  </span>
                </h1>

                <p className="text-lg text-white/70 max-w-md leading-relaxed">
                  Manage your uploads, track progress, and continue your journey with our secure platform.
                </p>
              </motion.div>

              {/* Decorative glowing lines */}
              <motion.div
                className="w-40 h-1 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full shadow-lg shadow-blue-500/40"
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: "10rem" }}
                transition={{ duration: 0.6, delay: 0.4 }}
              />
            </div>
          </motion.div>

          {/* RIGHT SIDE LOGIN FORM */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="bg-slate-900/80 backdrop-blur-xl border border-white/10 rounded-3xl p-10 shadow-2xl">
              <div className="text-center mb-8">
                <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-blue-500/30">
                  <Mail className="w-8 h-8 text-white" />
                </div>
                <h2 className="text-3xl font-bold text-white mb-2">Sign In</h2>
                <p className="text-white/50">Enter your credentials to continue</p>
              </div>

              <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-5">
                  <FormField
                    control={form.control}
                    name="username"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel className="text-white/80 font-medium">
                          Username or Email
                        </FormLabel>
                        <FormControl>
                          <div className="relative">
                            <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/40" />
                            <Input
                              placeholder="Enter your username"
                              className="h-12 pl-12 bg-white/5 border-white/10 text-white placeholder:text-white/30 focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 rounded-xl"
                              data-testid="input-username"
                              {...field}
                            />
                          </div>
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="password"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel className="text-white/80 font-medium">
                          Password
                        </FormLabel>
                        <FormControl>
                          <div className="relative">
                            <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/40" />
                            <Input
                              type={showPassword ? "text" : "password"}
                              placeholder="Enter your password"
                              className="h-12 pl-12 pr-12 bg-white/5 border-white/10 text-white placeholder:text-white/30 focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 rounded-xl"
                              data-testid="input-password"
                              {...field}
                            />
                            <button
                              type="button"
                              onClick={() => setShowPassword(!showPassword)}
                              className="absolute right-4 top-1/2 -translate-y-1/2 text-white/40 hover:text-white/60 transition-colors"
                            >
                              {showPassword ? (
                                <EyeOff className="w-5 h-5" />
                              ) : (
                                <Eye className="w-5 h-5" />
                              )}
                            </button>
                          </div>
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <div className="flex items-center justify-between pt-1">
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="remember"
                        checked={rememberMe}
                        onCheckedChange={(checked) => setRememberMe(checked as boolean)}
                        className="border-white/20 data-[state=checked]:bg-blue-500 data-[state=checked]:border-blue-500"
                        data-testid="checkbox-remember"
                      />
                      <label
                        htmlFor="remember"
                        className="text-sm text-white/60 cursor-pointer"
                      >
                        Remember me
                      </label>
                    </div>
                    <a
                      href="#"
                      className="text-sm text-blue-400 hover:text-blue-300 transition-colors"
                      data-testid="link-forgot-password"
                    >
                      Forgot password?
                    </a>
                  </div>

                  <Button
                    type="submit"
                    className="w-full h-12 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white font-semibold rounded-xl shadow-lg shadow-blue-500/25 transition-all mt-4"
                    disabled={loginMutation.isPending}
                    data-testid="button-login"
                  >
                    {loginMutation.isPending ? (
                      <span className="flex items-center gap-2">
                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Signing in...
                      </span>
                    ) : (
                      <span className="flex items-center gap-2">
                        Sign In
                        <ArrowRight className="w-5 h-5" />
                      </span>
                    )}
                  </Button>
                </form>
              </Form>

              {/* RIGHT SIDE SIGN UP LINK */}
              <div className="mt-10 pt-6 border-t border-white/10 text-center">
                <p className="text-white/50 text-sm">
                  Don't have an account?{" "}
                  <button
                    onClick={() => setLocation("/register")}
                    className="text-blue-400 hover:text-blue-300 font-medium transition-colors"
                  >
                    Create one for free
                  </button>
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
