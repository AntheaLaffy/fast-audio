#include "conv.h"
#include "conv2d.h"
#include <oneapi/dnnl/dnnl.hpp>
#include <omp.h>
#include <cstring>
#include <memory>
#include <new>

#if DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_OMP
#error "This adapter requires an OpenMP oneDNN CPU build"
#endif

using namespace dnnl;

// OpenMP ICVs belong to the calling task. Restore them so a plan does not
// change the caller's thread policy or the neighboring oneMKL operation.
struct Threads {
    int previous, dynamic;
    explicit Threads(int n) : previous(omp_get_max_threads()), dynamic(omp_get_dynamic()) {
        omp_set_dynamic(0);
        omp_set_num_threads(n);
    }
    ~Threads() { omp_set_num_threads(previous); omp_set_dynamic(dynamic); }
};

struct ConvPlan {
    int threads;
    engine cpu;
    stream queue;
    primitive_desc_base pd;
    primitive conv;
    memory user_src, user_dst, src, dst, weights, bias, scratch;
    reorder input_reorder, output_reorder;
    bool reorder_input = false, reorder_output = false, has_bias = false;
    size_t workspace_bytes = 0;
    size_t parameter_bytes = 0;
    const char *implementation = nullptr;
    explicit ConvPlan(int n) : threads(n), cpu(engine::kind::cpu, 0), queue(cpu) {}
};

static void execute(const primitive &op, const stream &queue, int count, const dnnl_exec_arg_t *args) {
    error::wrap_c_api(dnnl_primitive_execute(op.get(), queue.get(), count, args), "execute primitive");
}

static int create_conv(const memory::dims &xdims, const memory::dims &ydims,
                       const memory::dims &wdims, memory::format_tag data_format,
                       memory::format_tag weights_format, const memory::dims &strides,
                       const memory::dims &dilations, const memory::dims &pad_l,
                       const memory::dims &pad_r, bool transposed, const float *weights,
                       const float *bias, int relu, int threads, void **out) noexcept {
    try {
        Threads policy(threads);
        auto p = std::make_unique<ConvPlan>(threads);
        const auto fp32 = memory::data_type::f32;
        const auto any = memory::format_tag::any;
        const auto plain_src = memory::desc(xdims, fp32, data_format);
        const auto plain_dst = memory::desc(ydims, fp32, data_format);
        const auto plain_weights = memory::desc(wdims, fp32, weights_format);
        const auto bias_desc = bias ? memory::desc({ydims[1]}, fp32, memory::format_tag::x) : memory::desc();
        primitive_attr attr;
        attr.set_scratchpad_mode(scratchpad_mode::user);
        attr.set_fpmath_mode(fpmath_mode::strict);
        if (relu) {
            post_ops ops;
            ops.append_eltwise(algorithm::eltwise_relu, 0.0f, 0.0f);
            attr.set_post_ops(ops);
        }
        const auto src_any = memory::desc(xdims, fp32, any);
        const auto dst_any = memory::desc(ydims, fp32, any);
        const auto weights_any = memory::desc(wdims, fp32, any);
        if (transposed) {
            auto pd = deconvolution_forward::primitive_desc(p->cpu,prop_kind::forward_inference,
                algorithm::deconvolution_direct,src_any,weights_any,bias_desc,dst_any,
                strides,dilations,pad_l,pad_r,attr);
            p->pd = pd; p->conv = deconvolution_forward(pd);
        } else {
            auto pd = convolution_forward::primitive_desc(p->cpu,prop_kind::forward_inference,
                algorithm::convolution_direct,src_any,weights_any,bias_desc,dst_any,
                strides,dilations,pad_l,pad_r,attr);
            p->pd = pd; p->conv = convolution_forward(pd);
        }
        p->user_src = memory(plain_src, p->cpu, DNNL_MEMORY_NONE);
        p->user_dst = memory(plain_dst, p->cpu, DNNL_MEMORY_NONE);
        p->reorder_input = p->pd.src_desc() != plain_src;
        p->reorder_output = p->pd.dst_desc() != plain_dst;
        p->src = p->reorder_input ? memory(p->pd.src_desc(), p->cpu) : p->user_src;
        p->dst = p->reorder_output ? memory(p->pd.dst_desc(), p->cpu) : p->user_dst;
        if (p->reorder_input) {
            p->input_reorder = reorder(p->user_src, p->src);
            p->workspace_bytes += p->pd.src_desc().get_size();
        }
        if (p->reorder_output) {
            p->output_reorder = reorder(p->dst, p->user_dst);
            p->workspace_bytes += p->pd.dst_desc().get_size();
        }
        p->scratch = memory(p->pd.scratchpad_desc(), p->cpu);
        p->workspace_bytes += p->pd.scratchpad_desc().get_size();
        p->weights = memory(p->pd.weights_desc(), p->cpu);
        auto source_weights = memory(plain_weights, p->cpu, const_cast<float *>(weights));
        auto pack = reorder(source_weights, p->weights);
        const dnnl_exec_arg_t pack_args[] = {{DNNL_ARG_FROM, source_weights.get()}, {DNNL_ARG_TO, p->weights.get()}};
        execute(pack, p->queue, 2, pack_args);
        p->has_bias = bias != nullptr;
        if (bias) {
            p->bias = memory(bias_desc, p->cpu);
            std::memcpy(p->bias.get_data_handle(), bias, (size_t)ydims[1]*sizeof(float));
        }
        p->queue.wait();
        p->parameter_bytes = p->pd.weights_desc().get_size() + (bias ? p->pd.weights_desc(1).get_size() : 0);
        p->implementation = p->pd.impl_info_str();
        *out = p.release();
        return dnnl_success;
    } catch (const error &e) { return e.status; }
      catch (const std::bad_alloc &) { return dnnl_out_of_memory; }
      catch (...) { return dnnl_runtime_error; }
}

extern "C" int fa_onednn_conv_run(ConvPlan *p, const float *input, float *output) noexcept {
    try {
        Threads policy(p->threads);
        p->user_src.set_data_handle(const_cast<float *>(input));
        p->user_dst.set_data_handle(output);
        if (p->reorder_input) {
            const dnnl_exec_arg_t args[] = {{DNNL_ARG_FROM, p->user_src.get()}, {DNNL_ARG_TO, p->src.get()}};
            execute(p->input_reorder, p->queue, 2, args);
        }
        const dnnl_exec_arg_t args[] = {{DNNL_ARG_SRC, p->src.get()}, {DNNL_ARG_WEIGHTS, p->weights.get()},
            {DNNL_ARG_DST, p->dst.get()}, {DNNL_ARG_SCRATCHPAD, p->scratch.get()}, {DNNL_ARG_BIAS, p->bias.get(true)}};
        execute(p->conv, p->queue, p->has_bias ? 5 : 4, args);
        if (p->reorder_output) {
            const dnnl_exec_arg_t args[] = {{DNNL_ARG_FROM, p->dst.get()}, {DNNL_ARG_TO, p->user_dst.get()}};
            execute(p->output_reorder, p->queue, 2, args);
        }
        p->queue.wait();
        return dnnl_success;
    } catch (const error &e) { return e.status; }
      catch (const std::bad_alloc &) { return dnnl_out_of_memory; }
      catch (...) { return dnnl_runtime_error; }
}
extern "C" void fa_onednn_conv_info(ConvPlan *p, size_t *workspace, size_t *weights, const char **impl) noexcept {
    *workspace = p->workspace_bytes;
    *weights = p->parameter_bytes;
    *impl = p->implementation;
}
extern "C" void fa_onednn_conv_destroy(ConvPlan *p) noexcept { delete p; }

extern "C" int fa_onednn_conv_create(ConvDims s, const float *weights, const float *bias,
                                      int relu, int threads, void **out) noexcept {
    try {
    const memory::dims wdims = s.groups==1 ? memory::dims{s.out_channels,s.in_channels,s.kernel}
        : memory::dims{s.groups,s.out_channels/s.groups,s.in_channels/s.groups,s.kernel};
    return create_conv({s.batch,s.in_channels,s.input_len},{s.batch,s.out_channels,s.output_len},
        wdims,memory::format_tag::ncw,s.groups==1 ? memory::format_tag::oiw : memory::format_tag::goiw,
        {s.stride},{s.dilation-1},{s.padding},{s.padding},false,weights,bias,relu,threads,out);
    } catch (const error &e) { return e.status; }
      catch (const std::bad_alloc &) { return dnnl_out_of_memory; }
      catch (...) { return dnnl_runtime_error; }
}
extern "C" int fa_onednn_conv2d_create(Conv2dDims s, const float *weights, const float *bias,
                                        int relu, int threads, void **out) noexcept {
    try {
    const memory::dims wdims = s.groups==1 ? memory::dims{s.out_channels,s.in_channels,s.kh,s.kw}
        : memory::dims{s.groups,s.out_channels/s.groups,s.in_channels/s.groups,s.kh,s.kw};
    const auto format = s.transposed ? (s.groups==1 ? memory::format_tag::iohw : memory::format_tag::giohw)
        : (s.groups==1 ? memory::format_tag::oihw : memory::format_tag::goihw);
    return create_conv({s.batch,s.in_channels,s.ih,s.iw},{s.batch,s.out_channels,s.oh,s.ow},
        wdims,memory::format_tag::nchw,format,{s.sh,s.sw},{s.dh-1,s.dw-1},
        {s.ph,s.pw},{s.ph-s.oph,s.pw-s.opw},s.transposed!=0,weights,bias,relu,threads,out);
    } catch (const error &e) { return e.status; }
      catch (const std::bad_alloc &) { return dnnl_out_of_memory; }
      catch (...) { return dnnl_runtime_error; }
}
